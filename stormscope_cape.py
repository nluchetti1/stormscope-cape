#!/usr/bin/env python3
"""
StormScope (3 km / 10 min) ensemble nowcast over a Cape Canaveral window.

Manually triggered only. Nothing in this script schedules, polls the clock,
or initiates a forecast on its own -- a run happens when you invoke `run`,
or when you drop a job file with `fire` while a `serve` daemon is watching.

Three subcommands
-----------------
  run    one-shot: load models, run one ensemble forecast, exit.
         Cold start costs ~2-4 min of checkpoint loading each time.

  serve  load models once and watch TRIGGER_DIR for job files. Fires in
         seconds. Costs full GPU rate while idle -- only worth it when
         you are actively working a case.

  fire   write a job file for a running `serve` daemon and return
         immediately. Does nothing if no daemon is running.

Ensembles
---------
Members are independent diffusion noise draws from the same initial
condition. --members and --batch-size are decoupled: members are run in
ceil(members/batch_size) sequential batches, so ensemble size is limited
by patience rather than VRAM.

Examples
--------
  python stormscope_cape.py run --init 2025-07-16T19:00 \
      --lead-minutes 120 --members 8 --batch-size 2

  python stormscope_cape.py serve --batch-size 2
  python stormscope_cape.py fire --init 2025-07-16T19:00 --members 8
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import xarray as xr
from tqdm import trange

from earth2studio.data import GOES, MRMS, GOESGLMGrid, fetch_data
from earth2studio.models.px.stormscope import (
    StormScopeBase,
    StormScopeGOES,
    StormScopeMRMS,
)

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
CAPE_LAT = 28.585           # LC-39A / SLC-40 area
CAPE_LON = -80.650

NMI_KM = 1.852
PAD_RADII_NMI = (5.0, 10.0)  # rings to summarize probabilities over

REFC_THRESH = 35.0           # dBZ, for probability products

# Forecast GLM flash density threshold, in raw event counts per cell.
#
# NOT zero. StormScope's MRMS head is a diffusion model, so glm_density is
# almost never exactly 0 -- it carries a low positive floor across the whole
# domain. Thresholding at >0 therefore marks nearly every grid point as
# "lightning" and drives the pad-ring probabilities to 1.00 regardless of
# the weather. Override with --glm-thresh; run with --glm-stats to print the
# actual distribution for a case and pick a defensible value.
GLM_THRESH = 1.0

GOES19_START = datetime(2025, 4, 7, tzinfo=timezone.utc)

WS = os.environ.get("STORMSCOPE_WS", "/workspace")
TRIGGER_DIR = os.path.join(WS, "triggers")
DONE_DIR = os.path.join(TRIGGER_DIR, "done")
FAIL_DIR = os.path.join(TRIGGER_DIR, "failed")
OUT_DIR = os.path.join(WS, "outputs")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def pick_satellite(init: datetime) -> str:
    """Operational GOES-East platform for an init time.

    earth2studio bounds goes16 at 2025-04-07 and will raise past it.
    """
    return "goes19" if init >= GOES19_START else "goes16"


def floor_to_10min(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc)
    return dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)


def probe_latest_init(latency_minutes=8, max_back_min=90, verbose=True):
    """Walk back in 10-min steps until an init has all its input frames.

    The model needs six frames at t-50..t-0. GOES ABI L2 lands on S3
    roughly 1-3 min after scan end, MRMS 1-2 min, and GLM once its 5-min
    bin closes. Rather than guess a fixed lag, probe the two frames most
    likely to be missing -- t-0 and t-50 -- and take the newest init where
    both resolve. Falls back to a fixed offset if probing fails outright.
    """
    from earth2studio.data import GOES, MRMS

    now = datetime.now(timezone.utc)
    t = floor_to_10min(now - timedelta(minutes=latency_minutes))
    for _ in range(max_back_min // 10 + 1):
        sat = pick_satellite(t)
        try:
            g = GOES(satellite=sat, scan_mode="C")
            m = MRMS()
            for probe in (t, t - timedelta(minutes=50)):
                pn = np.datetime64(probe.replace(tzinfo=None))
                g(time=pn, variable=["abi13c"])
                m(time=pn, variable=["refc"])
            if verbose:
                age = (now - t).total_seconds() / 60.0
                print(f"  latest usable init: {t:%Y-%m-%d %H:%M}Z "
                      f"({age:.0f} min old, {sat})")
            return t
        except Exception as e:
            if verbose:
                print(f"  {t:%H:%M}Z not ready ({type(e).__name__}), "
                      f"stepping back 10 min")
            t -= timedelta(minutes=10)
    raise SystemExit(
        f"No usable init found in the last {max_back_min} min. Check S3 "
        f"access, or pass an explicit --init.")


def parse_init(s: str | None, latency_minutes: int = 25) -> datetime:
    """Parse an init string, or pick the most recent likely-available one.

    The model needs 6 frames at t-50..t-0, so 'now' is never a valid init.
    GOES ABI L2 and MRMS usually land on S3 within a few minutes of valid
    time; back off further if you see 404s.
    """
    if s and str(s).strip().lower() in ("latest", "now", "auto"):
        return probe_latest_init(latency_minutes=max(latency_minutes, 6))
    if s:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return floor_to_10min(dt)
    return floor_to_10min(datetime.now(timezone.utc) - timedelta(minutes=latency_minutes))


def great_circle_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2.0) ** 2)
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def cape_window(lat2d, lon2d, center_lat, center_lon, half_km, dx_km=3.0):
    """Index slices for a square window centered on the nearest grid point.

    Searches by great-circle distance rather than assuming a regular mesh,
    because the HRRR grid is curvilinear Lambert Conformal.
    """
    lon2d = ((np.asarray(lon2d) + 180.0) % 360.0) - 180.0
    lat2d = np.asarray(lat2d)
    d = great_circle_km(lat2d, lon2d, center_lat, center_lon)
    j0, i0 = np.unravel_index(np.argmin(d), d.shape)
    half = int(round(half_km / dx_km))
    ny, nx = lat2d.shape
    ys = slice(max(0, j0 - half), min(ny, j0 + half + 1))
    xs = slice(max(0, i0 - half), min(nx, i0 + half + 1))
    return ys, xs, (j0, i0), float(d[j0, i0])


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------
class StormScopeRunner:
    """Holds the loaded models so `serve` can fire repeatedly without reloading."""

    def __init__(self, half_km=250.0, num_steps=100, s_churn=10.0,
                 compile_model=False, device=None, glm_thresh=GLM_THRESH,
                 glm_stats=False):
        self.half_km = half_km
        self.glm_thresh = glm_thresh
        self.glm_stats = glm_stats
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        t0 = time.time()
        print("loading StormScope package...", flush=True)
        package = StormScopeBase.load_default_package()
        sampler_args = {"num_steps": num_steps, "S_churn": s_churn}

        # GOES model: pure-obs, no external conditioning source needed.
        self.model = StormScopeGOES.load_model(
            package=package,
            conditioning_data_source=None,
            model_name="3km_10min",
            amp=True,
            compile=compile_model,
        ).to(self.device)
        self.model.eval()
        self.model.sampler_args = sampler_args

        # MRMS+GLM model is conditioned on the GOES forecast at each step.
        # Its conditioning source is swapped per-run to match the satellite.
        self.model_mrms = StormScopeMRMS.load_model(
            package=package,
            conditioning_data_source=GOES(satellite="goes19", scan_mode="C"),
            glm_data_source=GOESGLMGrid(satellite="east"),
            model_name="3km_10min",
            amp=True,
            compile=compile_model,
        ).to(self.device)
        self.model_mrms.eval()
        self.model_mrms.sampler_args = sampler_args

        print(f"models loaded in {time.time() - t0:.1f} s", flush=True)

        self.lat = self.model.latitudes.detach().cpu().numpy()
        self.lon = self.model.longitudes.detach().cpu().numpy()
        self.ys, self.xs, nearest, dist = cape_window(
            self.lat, self.lon, CAPE_LAT, CAPE_LON, half_km)

        self.lat_c = self.lat[self.ys, self.xs]
        self.lon_c = ((self.lon[self.ys, self.xs] + 180.0) % 360.0) - 180.0
        print(f"model grid : {self.lat.shape[0]} x {self.lat.shape[1]}")
        print(f"cape window: {self.lat_c.shape[0]} x {self.lat_c.shape[1]} "
              f"(nearest pt {dist:.2f} km from pad)")

        # distance-from-pad field, reused for the ring products
        self.pad_dist_km = great_circle_km(
            self.lat_c, self.lon_c, CAPE_LAT, CAPE_LON)
        self.ring_masks = {
            r: (self.pad_dist_km <= r * NMI_KM) for r in PAD_RADII_NMI
        }

        # interpolators depend only on static grids -- built once, cached
        self._goes_interp_sat = None
        self._mrms_interp_built = False
        self.mrms_src = MRMS()

    # -- initial conditions --------------------------------------------------
    def fetch_ic(self, init: datetime):
        """Fetch GOES + MRMS + GLM initial state for one init. Batch dim = 1."""
        satellite = pick_satellite(init)
        start_date = [np.datetime64(init.replace(tzinfo=None))]
        dev = self.device

        goes_src = GOES(satellite=satellite, scan_mode="C")
        goes_lat, goes_lon = GOES.grid(satellite=satellite, scan_mode="C")

        # Rebuild GOES interpolators only when the platform changes.
        if self._goes_interp_sat != satellite:
            print(f"  building GOES interpolators for {satellite}...", flush=True)
            self.model.build_input_interpolator(goes_lat, goes_lon)
            self.model_mrms.build_conditioning_interpolator(goes_lat, goes_lon)
            self.model_mrms.conditioning_data_source = goes_src
            self._goes_interp_sat = satellite

        in_coords = self.model.input_coords()
        x, x_coords = fetch_data(
            goes_src,
            time=start_date,
            variable=np.array(in_coords["variable"]),
            lead_time=in_coords["lead_time"],
            device=dev,
        )

        mrms_in = self.model_mrms.input_coords()
        radar_vars = np.array([
            v for v in self.model_mrms.variables
            if v not in set(self.model_mrms.glm_variables)
        ])
        x_radar, xc_radar = fetch_data(
            self.mrms_src,
            time=start_date,
            variable=radar_vars,
            lead_time=mrms_in["lead_time"],
            device=dev,
        )

        if not self._mrms_interp_built:
            self.model_mrms.build_input_interpolator(xc_radar["lat"], xc_radar["lon"])
            self._mrms_interp_built = True
        x_radar = self.model_mrms.input_interp(x_radar)

        glm_coords = mrms_in.copy()
        glm_coords["time"] = np.array(start_date)
        x_glm, _ = self.model_mrms.fetch_glm(glm_coords, device=dev)

        x_mrms = torch.cat([x_radar, x_glm], dim=2).to(dtype=torch.float32)
        xc_mrms = xc_radar.copy()
        xc_mrms["variable"] = np.array(self.model_mrms.variables)
        del xc_mrms["lat"], xc_mrms["lon"]
        xc_mrms["y"] = self.model_mrms.y
        xc_mrms["x"] = self.model_mrms.x

        return (x.to(dtype=torch.float32), x_coords, x_mrms, xc_mrms, satellite)

    # -- one batch of members ------------------------------------------------
    def _rollout_batch(self, ic, n_steps, nmem, seed, save_goes, desc):
        x0, xc0, xm0, xcm0, _ = ic
        dev = self.device

        # Independent noise per member comes from the sampler; we only need
        # to replicate the (identical) initial condition across the batch.
        x = x0.unsqueeze(0).repeat(nmem, *([1] * x0.dim())).contiguous()
        xm = xm0.unsqueeze(0).repeat(nmem, *([1] * xm0.dim())).contiguous()

        xc = xc0.copy()
        xc["batch"] = np.arange(nmem)
        xc.move_to_end("batch", last=False)
        xcm = xcm0.copy()
        xcm["batch"] = np.arange(nmem)
        xcm.move_to_end("batch", last=False)

        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        y, y_coords = x, xc
        ym, ym_coords = xm, xcm
        mrms_frames, goes_frames = [], []

        for step in trange(n_steps, desc=desc, leave=False):
            y_pred, y_pred_coords = self.model(y, y_coords)
            ym_pred, ym_pred_coords = self.model_mrms.call_with_conditioning(
                ym, ym_coords, conditioning=y, conditioning_coords=y_coords
            )

            mp = torch.where(self.model_mrms.valid_mask, ym_pred, torch.nan)
            mrms_frames.append(
                mp[..., self.ys, self.xs].detach().float().cpu().numpy())
            if save_goes:
                gp = torch.where(self.model.valid_mask, y_pred, torch.nan)
                goes_frames.append(
                    gp[..., self.ys, self.xs].detach().float().cpu().numpy())

            y, y_coords = self.model.next_input(
                y_pred, y_pred_coords, y, y_coords)
            ym, ym_coords = self.model_mrms.next_input(
                ym_pred, ym_pred_coords, ym, ym_coords)

        # frames are [B, T, L, C, y, x] with T = L = 1
        mrms_arr = np.stack([f[:, 0, 0] for f in mrms_frames], axis=1)
        goes_arr = (np.stack([f[:, 0, 0] for f in goes_frames], axis=1)
                    if save_goes else None)
        return mrms_arr, goes_arr

    # -- full ensemble -------------------------------------------------------
    def run(self, init, lead_minutes=120, members=8, batch_size=1,
            seed=0, save_goes=False):
        n_steps = int(np.ceil(lead_minutes / 10.0))
        nbatch = int(np.ceil(members / batch_size))

        print(f"\ninit    : {init:%Y-%m-%d %H:%M} UTC")
        print(f"lead    : {n_steps} x 10 min = {n_steps * 10} min")
        print(f"members : {members} in {nbatch} batch(es) of <= {batch_size}")

        run_started = datetime.now(timezone.utc)
        t_fetch = time.time()
        ic = self.fetch_ic(init)
        satellite = ic[4]
        print(f"  IC fetched in {time.time() - t_fetch:.1f} s ({satellite})")

        mrms_parts, goes_parts = [], []
        t_run = time.time()
        done = 0
        for b in range(nbatch):
            nmem = min(batch_size, members - done)
            t_b = time.time()
            m_arr, g_arr = self._rollout_batch(
                ic, n_steps, nmem, seed + 1000 * b, save_goes,
                desc=f"batch {b + 1}/{nbatch}")
            mrms_parts.append(m_arr)
            if save_goes:
                goes_parts.append(g_arr)
            done += nmem
            el = time.time() - t_b
            print(f"  batch {b + 1}/{nbatch}: {nmem} member(s), {el:.1f} s "
                  f"({el / max(n_steps, 1):.1f} s/step)")

        mrms_arr = np.concatenate(mrms_parts, axis=0)   # [M, step, C, y, x]
        goes_arr = np.concatenate(goes_parts, axis=0) if save_goes else None
        print(f"rollout total: {time.time() - t_run:.1f} s")

        valid_times = np.array([
            np.datetime64((init + timedelta(minutes=10 * (k + 1))).replace(tzinfo=None))
            for k in range(n_steps)
        ])
        ds = self._build_dataset(mrms_arr, goes_arr, valid_times, init,
                                 satellite, members, seed)
        run_done = datetime.now(timezone.utc)
        ds.attrs["run_started"] = run_started.isoformat(timespec="seconds")
        ds.attrs["run_completed"] = run_done.isoformat(timespec="seconds")
        ds.attrs["runtime_sec"] = round(
            (run_done - run_started).total_seconds(), 1)
        ds.attrs["run_kind"] = "ensemble" if members > 1 else "deterministic"

        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(
            OUT_DIR, f"stormscope_cape_{init:%Y%m%dT%H%M}Z_ens{members:02d}.nc")
        comp = {"zlib": True, "complevel": 4}
        ds.to_netcdf(out, encoding={v: comp for v in ds.data_vars})
        print(f"wrote {out}")
        print(f"  init {init:%H:%M}Z | completed {run_done:%H:%M:%S}Z | "
              f"{ds.attrs['runtime_sec']/60:.1f} min | "
              f"{ds.attrs['run_kind']}")

        self._print_pad_table(ds)
        return out

    # -- assemble dataset + derived products ---------------------------------
    def _build_dataset(self, mrms_arr, goes_arr, valid_times, init,
                       satellite, members, seed):
        ch = list(self.model_mrms.variables)
        refc = mrms_arr[:, :, ch.index("refc")]          # [M, step, y, x]
        glm = mrms_arr[:, :, ch.index("glm_density")]

        with np.errstate(invalid="ignore"):
            p_refc = np.nanmean((refc >= REFC_THRESH).astype("float32"), axis=0)
            p_light = np.nanmean((glm > self.glm_thresh).astype("float32"), axis=0)
            ens_mean = np.nanmean(refc, axis=0)
            ens_max = np.nanmax(refc, axis=0)

        data_vars = {
            "mrms": (("member", "valid_time", "mrms_channel", "y", "x"),
                     mrms_arr.astype("float32")),
            f"p_refc_ge_{int(REFC_THRESH)}": (("valid_time", "y", "x"), p_refc),
            "p_lightning": (("valid_time", "y", "x"), p_light),
            "refc_ens_mean": (("valid_time", "y", "x"), ens_mean.astype("float32")),
            "refc_ens_max": (("valid_time", "y", "x"), ens_max.astype("float32")),
        }
        if goes_arr is not None:
            data_vars["goes"] = (
                ("member", "valid_time", "goes_channel", "y", "x"),
                goes_arr.astype("float32"))

        # ring probabilities: per member, does ANY point inside the ring
        # exceed threshold at this lead time -> fraction of members
        for r, mask in self.ring_masks.items():
            tag = f"{int(r)}nmi"
            with np.errstate(invalid="ignore"):
                hit_l = np.nanmax(np.where(mask, glm, np.nan), axis=(2, 3)) > self.glm_thresh
                hit_r = np.nanmax(np.where(mask, refc, np.nan), axis=(2, 3)) >= REFC_THRESH
            data_vars[f"p_lightning_{tag}"] = (
                ("valid_time",), hit_l.mean(axis=0).astype("float32"))
            data_vars[f"p_refc_{tag}"] = (
                ("valid_time",), hit_r.mean(axis=0).astype("float32"))

        coords = {
            "member": np.arange(members),
            "valid_time": valid_times,
            "mrms_channel": np.array(ch),
            "lat": (("y", "x"), self.lat_c),
            "lon": (("y", "x"), self.lon_c),
            "pad_dist_km": (("y", "x"), self.pad_dist_km.astype("float32")),
        }
        if goes_arr is not None:
            coords["goes_channel"] = np.array(self.model.variables)

        if self.glm_stats:
            g = glm[np.isfinite(glm)]
            print("\nForecast glm_density distribution (all members/times/points)")
            print(f"  min {g.min():.4g}  max {g.max():.4g}  mean {g.mean():.4g}")
            for q in (50, 90, 99, 99.9, 99.99):
                print(f"  p{q:<6}: {np.percentile(g, q):.4g}")
            for thr in (0.0, 0.1, 0.5, 1.0, 5.0, 10.0):
                print(f"  fraction of points > {thr:<5}: "
                      f"{float((g > thr).mean()):.5f}")
            print(f"  (current --glm-thresh = {self.glm_thresh})")

        return xr.Dataset(
            data_vars=data_vars, coords=coords,
            attrs={
                "model": "NVIDIA StormScope 3km_10min (GOES + MRMS/GLM)",
                "init_time": init.isoformat(),
                "satellite": satellite,
                "members": members,
                "seed": seed,
                "refc_threshold_dbz": REFC_THRESH,
                "glm_threshold": self.glm_thresh,
                "crop_center_lat": CAPE_LAT,
                "crop_center_lon": CAPE_LON,
                "crop_half_km": self.half_km,
                "note": ("Research nowcast guidance. Not an LLCC evaluation; "
                         "encodes no launch commit criteria."),
            },
        )

    def _print_pad_table(self, ds):
        print("\nProbability of occurrence within radius of pad (ensemble fraction)")
        hdr = "lead   valid_time      "
        for r in PAD_RADII_NMI:
            hdr += f" lgt<{int(r)}nmi  refc<{int(r)}nmi"
        print(hdr)
        print("-" * len(hdr))
        t0 = ds.valid_time.values[0] - np.timedelta64(10, "m")
        for t in ds.valid_time.values:
            lead = int((t - t0) / np.timedelta64(1, "m"))
            row = f"{lead:4d}m  {str(t)[:16]}  "
            for r in PAD_RADII_NMI:
                tag = f"{int(r)}nmi"
                row += (f"   {float(ds[f'p_lightning_{tag}'].sel(valid_time=t)):5.2f}"
                        f"      {float(ds[f'p_refc_{tag}'].sel(valid_time=t)):5.2f}")
            print(row)


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
def cmd_run(args):
    runner = StormScopeRunner(half_km=args.half_km, num_steps=args.num_steps,
                              compile_model=args.compile,
                              glm_thresh=args.glm_thresh,
                              glm_stats=args.glm_stats)
    runner.run(parse_init(args.init, args.latency_min), lead_minutes=args.lead_minutes,
               members=args.members, batch_size=args.batch_size,
               seed=args.seed, save_goes=args.save_goes)


def cmd_fire(args):
    """Write a job file. Does nothing unless a `serve` daemon is watching."""
    os.makedirs(TRIGGER_DIR, exist_ok=True)
    job = {
        "init": parse_init(args.init, args.latency_min).isoformat(),
        "lead_minutes": args.lead_minutes,
        "members": args.members,
        "seed": args.seed,
        "save_goes": args.save_goes,
        "glm_thresh": args.glm_thresh,
        "submitted": datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(
        TRIGGER_DIR,
        f"job_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(job, f, indent=2)
    os.rename(tmp, path)   # atomic, so the daemon never sees a partial file

    print(f"queued {path}")
    print(json.dumps(job, indent=2))
    if not glob.glob(os.path.join(WS, ".serve_alive")):
        print("\nNOTE: no serve daemon appears to be running. This job will "
              "sit here until one starts.")


def cmd_serve(args):
    os.makedirs(TRIGGER_DIR, exist_ok=True)
    os.makedirs(DONE_DIR, exist_ok=True)
    os.makedirs(FAIL_DIR, exist_ok=True)

    runner = StormScopeRunner(half_km=args.half_km, num_steps=args.num_steps,
                              compile_model=args.compile)

    alive = os.path.join(WS, ".serve_alive")
    with open(alive, "w") as f:
        f.write(str(os.getpid()))

    print(f"\nwatching {TRIGGER_DIR} (poll {args.poll}s). Ctrl-C to stop.")
    print("nothing runs until a job file appears here.\n", flush=True)

    try:
        while True:
            jobs = sorted(glob.glob(os.path.join(TRIGGER_DIR, "job_*.json")))
            if not jobs:
                time.sleep(args.poll)
                continue

            path = jobs[0]
            try:
                with open(path) as f:
                    job = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"unreadable job {path}: {e}")
                shutil.move(path, os.path.join(FAIL_DIR, os.path.basename(path)))
                continue

            print(f"\n=== job {os.path.basename(path)} ===")
            runner.glm_thresh = job.get("glm_thresh", runner.glm_thresh)
            try:
                runner.run(
                    parse_init(job["init"]),
                    lead_minutes=job.get("lead_minutes", 120),
                    members=job.get("members", 8),
                    batch_size=args.batch_size,
                    seed=job.get("seed", 0),
                    save_goes=job.get("save_goes", False),
                )
                shutil.move(path, os.path.join(DONE_DIR, os.path.basename(path)))
            except Exception as e:
                print(f"job FAILED: {type(e).__name__}: {e}", file=sys.stderr)
                shutil.move(path, os.path.join(FAIL_DIR, os.path.basename(path)))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print("idle, waiting for next job\n", flush=True)
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        if os.path.exists(alive):
            os.remove(alive)


def cmd_cycle(args):
    """Run a forecast every N minutes for a fixed window, then stop.

    You start this once. It loads the models once and reuses them, so each
    cycle pays only the IC fetch and the rollout -- not the 2-4 min
    checkpoint load. It stops on its own after --duration-hours.

    If a cycle overruns its slot the next one is SKIPPED rather than
    queued, so the loop stays anchored to wall-clock time instead of
    drifting further behind with every iteration.
    """
    started = datetime.now(timezone.utc)
    finish_by = started + timedelta(hours=args.duration_hours)

    print("=" * 62)
    print(f" cycling every {args.every} min until "
          f"{finish_by:%Y-%m-%d %H:%M}Z ({args.duration_hours} h)")
    print(f" {args.members} members, {args.lead_minutes} min lead, "
          f"batch {args.batch_size}")
    if args.ensemble_every and args.ensemble_members > args.members:
        _el = args.ensemble_lead_minutes or args.lead_minutes
        print(f" tiered: {args.members} member / {args.lead_minutes} min lead "
              f"every {args.every:.0f} min, "
              f"{args.ensemble_members} members / {_el} min lead every "
              f"{args.ensemble_every:.0f} min (on the clock)")
        if args.per_member_min:
            # per_member_min is quoted at --lead-minutes; scale the
            # ensemble if it runs a different lead.
            ens_lead = args.ensemble_lead_minutes or args.lead_minutes
            scale = ens_lead / float(args.lead_minutes)
            det = args.members * args.per_member_min
            ens = args.ensemble_members * args.per_member_min * scale
            per_window = args.ensemble_every
            n_det_slots = max(int(per_window / args.every) - 1, 0)
            used = ens + n_det_slots * det
            duty = 100.0 * used / per_window
            print(f" estimated per {per_window:.0f} min window: "
                  f"ensemble {ens:.0f} min + {n_det_slots} x {det:.0f} min "
                  f"deterministic = {used:.0f} min ({duty:.0f}% duty)")
            if ens > per_window:
                fits = max(int(per_window // args.per_member_min), 1)
                print(f"   WARNING: the ensemble alone ({ens:.0f} min) exceeds "
                      f"its {per_window:.0f} min window. It will run "
                      f"back-to-back and NO deterministic runs will happen.")
                print(f"   Fix: --ensemble-members {fits}, or "
                      f"--ensemble-every {int(ens // 30 + 1) * 30}, or halve "
                      f"the cost with --num-steps 50.")
            elif used > per_window:
                print(f"   WARNING: ensemble leaves only "
                      f"{per_window - ens:.0f} min for {n_det_slots} "
                      f"deterministic run(s) needing {n_det_slots * det:.0f} "
                      f"min. Some slots will be skipped.")
            elif duty > 85:
                print(f"   NOTE: {duty:.0f}% duty leaves little slack for "
                      f"post-processing. Consider --num-steps 50.")
    if args.post_run:
        print(f" post-run: {args.post_run}")
    print(f" stop the loop any time with Ctrl-C")
    print("=" * 62, flush=True)

    runner = StormScopeRunner(half_km=args.half_km, num_steps=args.num_steps,
                              compile_model=args.compile,
                              glm_thresh=args.glm_thresh)

    n_ok = n_fail = n_skip = 0
    last_init = None
    try:
        while datetime.now(timezone.utc) < finish_by:
            cycle_start = datetime.now(timezone.utc)
            try:
                init = probe_latest_init(latency_minutes=args.latency_min)
            except SystemExit as e:
                print(f"  {e}\n  waiting {args.every} min and retrying",
                      flush=True)
                _sleep_to_next(args.every, finish_by)
                continue

            if init == last_init:
                print(f"  {init:%H:%M}Z already run, no new data yet -- "
                      f"skipping this slot", flush=True)
                n_skip += 1
                _sleep_to_next(args.every, finish_by)
                continue

            # Tiered schedule: a slot aligned to --ensemble-every gets the
            # full ensemble, every other slot gets the cheap deterministic
            # run. Alignment is to wall clock, not to loop start, so the
            # ensemble always lands on tidy times (:00, :30, ...).
            big = False
            if args.ensemble_every and args.ensemble_members > args.members:
                mins = init.hour * 60 + init.minute
                big = (mins % int(args.ensemble_every)) == 0
            n_mem = args.ensemble_members if big else args.members
            n_lead = (args.ensemble_lead_minutes or args.lead_minutes) if big \
                else args.lead_minutes
            kind = "ENSEMBLE" if big else "deterministic"

            print(f"\n--- cycle {n_ok + n_fail + 1} | "
                  f"{cycle_start:%H:%M:%S}Z | init {init:%H:%M}Z | "
                  f"{kind}, {n_mem} member(s), {n_lead} min lead ---",
                  flush=True)
            try:
                out = runner.run(init, lead_minutes=n_lead,
                                 members=n_mem,
                                 batch_size=min(args.batch_size, n_mem),
                                 seed=args.seed, save_goes=False)
                last_init = init
                n_ok += 1
                if args.post_run:
                    cmd = args.post_run.replace("{ncfile}", out)
                    print(f"  post-run: {cmd}", flush=True)
                    rc = os.system(cmd)
                    if rc != 0:
                        print(f"  post-run exited {rc >> 8} "
                              f"(forecast is still saved)")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                n_fail += 1
                print(f"  cycle FAILED: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds() / 60
            budget = args.ensemble_every if big else args.every
            if elapsed > budget:
                print(f"  NOTE: this {kind} cycle took {elapsed:.1f} min, over "
                      f"its {budget:.0f} min budget. Intermediate slots will "
                      f"be skipped. Reduce members or --num-steps, or widen "
                      f"the interval.", flush=True)
            _sleep_to_next(args.every, finish_by)

    except KeyboardInterrupt:
        print("\n  interrupted", flush=True)

    print(f"\n{'=' * 62}")
    print(f" done: {n_ok} ok, {n_fail} failed, {n_skip} skipped "
          f"over {(datetime.now(timezone.utc) - started).total_seconds()/3600:.2f} h")
    print("=" * 62, flush=True)

    if args.then_stop:
        pod = os.environ.get("RUNPOD_POD_ID", "")
        if not pod:
            print(" --then-stop set but RUNPOD_POD_ID is not in the "
                  "environment; stop the pod yourself.")
        else:
            print(f" stopping pod {pod} in 60 s -- Ctrl-C to cancel", flush=True)
            try:
                time.sleep(60)
                os.system(f"runpodctl stop pod {pod}")
            except KeyboardInterrupt:
                print(" cancelled; pod left running")


def _sleep_to_next(every_min, finish_by):
    """Sleep until the next slot boundary, or until the window closes."""
    now = datetime.now(timezone.utc)
    step = timedelta(minutes=every_min)
    epoch = now.replace(hour=0, minute=0, second=0, microsecond=0)
    n = int((now - epoch) / step) + 1
    nxt = min(epoch + n * step, finish_by)
    wait = (nxt - now).total_seconds()
    if wait > 0 and nxt < finish_by:
        print(f"  sleeping {wait/60:.1f} min until {nxt:%H:%M}Z", flush=True)
        time.sleep(wait)
    elif wait > 0:
        time.sleep(min(wait, 5))


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, with_batch=True):
        sp.add_argument("--half-km", type=float, default=250.0,
                        help="Half-width of the Cape crop box in km.")
        sp.add_argument("--num-steps", type=int, default=100,
                        help="EDM sampler steps. 30 is much faster, try both.")
        sp.add_argument("--compile", action="store_true",
                        help="torch.compile the models. Slow first step, "
                             "faster afterwards. Worth it only for serve.")
        if with_batch:
            sp.add_argument("--batch-size", type=int, default=1,
                            help="Members per GPU batch. VRAM scales ~linearly.")

    def jobargs(sp):
        sp.add_argument("--init", default=None,
                        help="Init time UTC ISO8601, or 'latest' to probe S3 "
                             "for the newest init with all six input frames "
                             "present. Default: most recent likely-available "
                             "10-min mark.")
        sp.add_argument("--latency-min", type=float, default=25.0,
                        help="Minutes to back off from now when no explicit "
                             "init is given. With --init latest this is the "
                             "starting point for the probe.")
        sp.add_argument("--lead-minutes", type=int, default=120)
        sp.add_argument("--members", type=int, default=8)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--save-goes", action="store_true",
                        help="Also store the 8 GOES channels (large).")
        sp.add_argument("--glm-thresh", type=float, default=GLM_THRESH,
                        help="Forecast GLM flash-density threshold for the "
                             "lightning probability products. Do NOT use 0 -- "
                             "the diffusion output has a positive floor "
                             "everywhere.")
        sp.add_argument("--glm-stats", action="store_true",
                        help="Print the forecast glm_density distribution so "
                             "you can choose a defensible threshold.")

    sp = sub.add_parser("run", help="one-shot ensemble forecast")
    common(sp)
    jobargs(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("serve", help="load models and watch for job files")
    common(sp)
    sp.add_argument("--poll", type=float, default=5.0,
                    help="Seconds between trigger-directory checks.")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser(
        "cycle",
        help="run a forecast every N minutes for a fixed window, then stop")
    common(sp)
    jobargs(sp)
    sp.add_argument("--every", type=float, default=10.0,
                    help="Minutes between cycles. Must exceed the time one "
                         "forecast takes or slots get skipped.")
    sp.add_argument("--duration-hours", type=float, default=6.0,
                    help="Stop after this many hours.")
    sp.add_argument("--ensemble-every", type=float, default=0.0,
                    help="Minutes between FULL ensemble runs, aligned to the "
                         "wall clock (30 -> :00 and :30). Other slots use "
                         "--members. 0 disables tiering.")
    sp.add_argument("--ensemble-members", type=int, default=8,
                    help="Members on an ensemble slot. Must fit inside "
                         "--ensemble-every at your measured per-member cost.")
    sp.add_argument("--ensemble-lead-minutes", type=int, default=0,
                    help="Lead time for ensemble slots, if different from "
                         "--lead-minutes. Runtime scales linearly with lead, "
                         "so a shorter ensemble is the cheapest way to make "
                         "a tight schedule fit. 0 = same as deterministic.")
    sp.add_argument("--post-run", default="",
                    help="Shell command after each successful forecast. "
                         "'{ncfile}' is replaced with the output path. "
                         "E.g. './deploy_pages.sh {ncfile}'")
    sp.add_argument("--per-member-min", type=float, default=0.0,
                    help="Your measured minutes per member. Used only to "
                         "sanity-check the schedule at startup.")
    sp.add_argument("--then-stop", action="store_true",
                    help="Stop the RunPod pod when the window closes. "
                         "Needs RUNPOD_POD_ID in the environment.")
    sp.set_defaults(func=cmd_cycle)

    sp = sub.add_parser("fire", help="queue a job for a running serve daemon")
    jobargs(sp)
    sp.set_defaults(func=cmd_fire)

    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    a.func(a)
