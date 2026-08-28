#!/usr/bin/env python3
"""
Score elapsed StormScope runs against observations and publish the result.

This is the deferred half of the workflow. export_viewer.py publishes the
forecast as soon as it is made; this pass comes back later, once the valid
times have elapsed, fetches observed MRMS and GLM, renders matching frames,
computes skill, and updates the run's manifest in place.

It is a QUEUE, not a fixed delay. Every run in the site that has not been
scored and whose last valid time is more than --min-age-min in the past
gets picked up. Run it daily at 14Z, or whenever you next have a box up --
it catches up on whatever it missed either way.

CPU only. No GPU needed, but earth2studio must be importable for the
observation fetch.

Usage
-----
  python verify_pending.py                       # everything pending
  python verify_pending.py --slug 20260826T1700Z # one run
  python verify_pending.py --dry-run             # just list what is pending

After it finishes, commit and push (or let deploy_pages.sh do it).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
from matplotlib import colors as mcolors

from export_viewer import (
    great_circle_km,
    grid_spacing_km,
    circular_footprint,
    neighborhood_probability,
    zoom_slices,
    render_frame,
    render_colorbar,
)


# ---------------------------------------------------------------------------
# skill metrics
# ---------------------------------------------------------------------------
def neighborhood_fraction(field2d, thresh, dx_km, radius_km):
    """Fraction of points within radius_km exceeding thresh, per grid point.

    This is the Po / Pf used by the fractions skill score.
    """
    from scipy import ndimage
    valid = np.isfinite(field2d)
    binary = (np.nan_to_num(field2d, nan=-np.inf) >= thresh).astype(np.float32)
    foot = circular_footprint(radius_km / dx_km)
    kern = foot.astype(np.float32) / float(foot.sum())
    frac = ndimage.convolve(binary, kern, mode="nearest")
    return np.where(valid, frac, np.nan)


def fractions_skill_score(pf, po):
    """FSS from forecast and observed neighborhood fractions.

    FSS = 1 - FBS / FBS_ref, where FBS is the mean squared difference of
    the fractions and FBS_ref is the worst-case reference. 1 is perfect,
    0 is no skill. Unlike point-by-point CSI this does not double-penalize
    a cell that is displaced but otherwise correct, which is the whole
    reason to use it on a generative nowcast.

    Returns NaN when neither field has any coverage -- an undefined score
    rather than a misleading 1.0 or 0.0.
    """
    m = np.isfinite(pf) & np.isfinite(po)
    if not m.any():
        return float("nan")
    a, b = pf[m], po[m]
    fbs = float(np.mean((a - b) ** 2))
    ref = float(np.mean(a ** 2) + np.mean(b ** 2))
    if ref == 0.0:
        return float("nan")
    return 1.0 - fbs / ref


def contingency(fcst, obs, thresh):
    m = np.isfinite(fcst) & np.isfinite(obs)
    f, o = fcst[m] >= thresh, obs[m] >= thresh
    h, mi, fa = np.sum(f & o), np.sum(~f & o), np.sum(f & ~o)
    pod = h / (h + mi) if (h + mi) else float("nan")
    far = fa / (h + fa) if (h + fa) else float("nan")
    csi = h / (h + mi + fa) if (h + mi + fa) else float("nan")
    bias = (h + fa) / (h + mi) if (h + mi) else float("nan")
    return pod, far, csi, bias


def brier(prob, obs_binary):
    """Brier score of a probability field against a binary observation."""
    m = np.isfinite(prob) & np.isfinite(obs_binary)
    if not m.any():
        return float("nan")
    return float(np.mean((prob[m] - obs_binary[m]) ** 2))


def clean(x):
    """JSON does not accept NaN; emit null instead."""
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


# ---------------------------------------------------------------------------
def find_nc(nc_dir, slug):
    hits = sorted(glob.glob(os.path.join(nc_dir, f"*{slug[:-1]}*_ens*.nc")))
    if not hits:
        hits = sorted(glob.glob(os.path.join(nc_dir, f"*{slug[:-1]}*.nc")))
    return hits[-1] if hits else None


def fetch_obs(valid_times, lat, lon):
    """Observed MRMS refc and gridded GLM density on the plot grid."""
    import torch
    from earth2studio.data import MRMS, GOESGLMGrid, fetch_data
    from earth2studio.utils.interp import NearestNeighborInterpolator

    dev = torch.device("cpu")
    tlat = torch.as_tensor(lat)
    tlon = torch.as_tensor(lon % 360.0)
    zero = np.array([np.timedelta64(0, "m")])

    mrms, glm_src = MRMS(), GOESGLMGrid(satellite="east")
    ir = ig = None
    refc, glm = [], []

    for t in valid_times:
        try:
            o, oc = fetch_data(mrms, time=[t], variable=np.array(["refc"]),
                               lead_time=zero, device=dev)
            if ir is None:
                ir = NearestNeighborInterpolator(
                    source_lats=oc["lat"], source_lons=oc["lon"],
                    target_lats=tlat, target_lons=tlon, max_dist_km=12.0)
            refc.append(ir(o).squeeze().detach().float().cpu().numpy())
        except Exception as e:
            print(f"    MRMS {str(t)[:16]}: {type(e).__name__}")
            refc.append(None)

        acc, oc = None, None
        for k in range(2):   # two 5-min bins ending at the valid time
            try:
                g, oc = fetch_data(glm_src, time=[t - np.timedelta64(5 * k, "m")],
                                   variable=np.array(["glm_density"]),
                                   lead_time=zero, device=dev)
                acc = g if acc is None else acc + g
            except Exception:
                pass
        if acc is None:
            print(f"    GLM  {str(t)[:16]}: unavailable")
            glm.append(None)
        else:
            if ig is None:
                ig = NearestNeighborInterpolator(
                    source_lats=oc["lat"], source_lons=oc["lon"],
                    target_lats=tlat, target_lons=tlon, max_dist_km=15.0)
            glm.append(ig(acc).squeeze().detach().float().cpu().numpy())

    return refc, glm


# ---------------------------------------------------------------------------
def verify_run(site_dir, slug, nc_dir, args):
    run_dir = os.path.join(site_dir, "runs", slug)
    mpath = os.path.join(run_dir, "manifest.json")
    with open(mpath) as f:
        man = json.load(f)

    nc = find_nc(nc_dir, slug)
    if not nc:
        print(f"  {slug}: no NetCDF found in {nc_dir}/ -- skipping")
        print(f"    (the .nc lives on the network volume; run this where "
              f"they are, or pass --nc-dir)")
        return None
    print(f"  {slug}: scoring against {os.path.basename(nc)}")

    ds = xr.open_dataset(nc)
    clat = ds.attrs.get("crop_center_lat", 28.585)
    clon = ds.attrs.get("crop_center_lon", -80.650)
    glm_thresh = float(ds.attrs.get("glm_threshold", 1.0))

    pad_d = (ds["pad_dist_km"].values if "pad_dist_km" in ds.coords
             else great_circle_km(ds["lat"].values, ds["lon"].values, clat, clon))
    zoom = args.zoom_km if args.zoom_km else None
    zy, zx = zoom_slices(pad_d, zoom) if zoom else (slice(None), slice(None))
    lat, lon = ds["lat"].values[zy, zx], ds["lon"].values[zy, zx]
    dx_km = grid_spacing_km(lat, lon)

    valid_times = ds["valid_time"].values
    refc_all = ds["mrms"].sel(mrms_channel="refc")
    glm_all = ds["mrms"].sel(mrms_channel="glm_density")

    obs_refc, obs_glm = fetch_obs(valid_times, lat, lon)
    if all(o is None for o in obs_refc):
        print("    no observations retrieved -- leaving run unverified")
        return None

    dbz_norm = mcolors.Normalize(vmin=5.0, vmax=60.0)
    gmax = max(float(np.nanmax(glm_all.values[..., zy, zx])), 2.0)
    glm_norm = mcolors.LogNorm(vmin=max(glm_thresh, 0.5), vmax=gmax)
    nb = args.nbhd_km

    # ---- observed frames ---------------------------------------------------
    new_vars = []
    for vid, label, src, cmap, norm, mask_at in [
        ("obs_refc", "Observed reflectivity (MRMS)", obs_refc,
         "turbo", dbz_norm, 0.0),
        ("obs_glm", "Observed GLM flash density", obs_glm,
         "magma", glm_norm, glm_thresh),
    ]:
        vdir = os.path.join(run_dir, vid)
        os.makedirs(vdir, exist_ok=True)
        files = []
        for k, ld in enumerate(man["leads"]):
            fn = f"f{ld['minutes']:03d}.png"
            d = src[k] if src[k] is not None else np.full(lat.shape, np.nan)
            render_frame(os.path.join(vdir, fn), lat, lon, d, cmap, norm, mask_at)
            files.append(f"{vid}/{fn}")
        render_colorbar(os.path.join(vdir, "colorbar.png"), cmap, norm, label)
        new_vars.append({"id": vid, "label": label,
                         "units": "dBZ" if "refc" in vid else "events/cell",
                         "group": "refc" if "refc" in vid else "glm",
                         "group_label": ("Reflectivity" if "refc" in vid
                                         else "GLM flash density"),
                         "variant": "obs", "variant_label": "Obs",
                         "colorbar": f"{vid}/colorbar.png", "frames": files,
                         "observed": True})
        print(f"    {vid:<10} {len(files)} frames")

    # ---- skill -------------------------------------------------------------
    rows = []
    for k, ld in enumerate(man["leads"]):
        row = {"minutes": ld["minutes"], "valid": ld["valid"]}

        if obs_refc[k] is not None:
            fstack = refc_all.isel(valid_time=k).values[..., zy, zx]
            o = obs_refc[k]
            pod, far, csi, bias = contingency(np.nanmean(fstack, axis=0),
                                              o, args.thresh)
            pf = np.nanmean([neighborhood_fraction(fstack[m], args.thresh,
                                                   dx_km, nb)
                             for m in range(fstack.shape[0])], axis=0)
            po = neighborhood_fraction(o, args.thresh, dx_km, nb)
            prob = neighborhood_probability(fstack, args.thresh, dx_km, nb,
                                            mode=args.nep_mode)
            obin = (neighborhood_probability(o[None], args.thresh, dx_km, nb,
                                             mode="nmep") > 0).astype(float)
            row.update(pod=clean(pod), far=clean(far), csi=clean(csi),
                       bias=clean(bias),
                       fss_refc=clean(fractions_skill_score(pf, po)),
                       brier_refc=clean(brier(prob, obin)),
                       obs_max_refc=clean(np.nanmax(o)))

        if obs_glm[k] is not None:
            gstack = glm_all.isel(valid_time=k).values[..., zy, zx]
            og = obs_glm[k]
            pf = np.nanmean([neighborhood_fraction(gstack[m], glm_thresh,
                                                   dx_km, nb)
                             for m in range(gstack.shape[0])], axis=0)
            po = neighborhood_fraction(og, glm_thresh, dx_km, nb)
            prob = neighborhood_probability(gstack, glm_thresh, dx_km, nb,
                                            mode=args.nep_mode)
            obin = (neighborhood_probability(og[None], glm_thresh, dx_km, nb,
                                             mode="nmep") > 0).astype(float)
            row.update(fss_glm=clean(fractions_skill_score(pf, po)),
                       brier_glm=clean(brier(prob, obin)),
                       obs_glm_cells=int(np.nansum(og > glm_thresh)))
        rows.append(row)

    man["variables"] = [v for v in man["variables"]
                        if not v.get("observed")] + new_vars
    man["verification"] = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "neighborhood_km": nb,
        "nep_mode": args.nep_mode,
        "refc_threshold": args.thresh,
        "glm_threshold": glm_thresh,
        "leads": rows,
        "caveat": ("FSS is the fair read for a generative nowcast; "
                   "point CSI double-penalizes displaced cells. GLM "
                   "scores depend on an uncalibrated flash-density "
                   "threshold."),
    }
    with open(mpath, "w") as f:
        json.dump(man, f, indent=2)

    print("    lead    CSI   FSS_refc  FSS_glm  Brier_refc")
    for r in rows:
        f = lambda k: ("  --  " if r.get(k) is None else f"{r[k]:6.3f}")
        print(f"    {r['minutes']:4d}m {f('csi')}   {f('fss_refc')}  "
              f"{f('fss_glm')}  {f('brier_refc')}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site-dir", default="site")
    ap.add_argument("--nc-dir", default="outputs",
                    help="Where the .nc files live (network volume).")
    ap.add_argument("--slug", default=None,
                    help="Verify one run instead of the whole pending queue.")
    ap.add_argument("--min-age-min", type=float, default=30.0,
                    help="Wait this long past the last valid time before "
                         "scoring, so observations have landed on S3.")
    ap.add_argument("--max-age-days", type=float, default=14.0,
                    help="Do not bother with runs older than this.")
    ap.add_argument("--thresh", type=float, default=35.0)
    ap.add_argument("--nbhd-km", type=float, default=10.0)
    ap.add_argument("--nep-mode", choices=["nmep", "nep"], default="nmep")
    ap.add_argument("--zoom-km", type=float, default=120.0)
    ap.add_argument("--force", action="store_true",
                    help="Re-verify runs that already have scores.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what is pending and exit.")
    args = ap.parse_args()

    idx_path = os.path.join(args.site_dir, "runs", "index.json")
    if not os.path.isfile(idx_path):
        raise SystemExit(f"No {idx_path}. Run export_viewer.py first.")
    with open(idx_path) as f:
        idx = json.load(f)

    now = datetime.now(timezone.utc)
    pending = []
    for entry in idx["runs"]:
        slug = entry["slug"]
        if args.slug and slug != args.slug:
            continue
        mpath = os.path.join(args.site_dir, "runs", slug, "manifest.json")
        if not os.path.isfile(mpath):
            continue
        with open(mpath) as f:
            man = json.load(f)
        if man.get("verification") and not args.force:
            continue

        init = datetime.fromisoformat(man["init"])
        last = init + timedelta(minutes=man["leads"][-1]["minutes"])
        age = (now - last).total_seconds() / 60.0
        if age < args.min_age_min:
            print(f"  {slug}: last valid time {abs(age):.0f} min "
                  f"{'ahead' if age < 0 else 'ago'} -- too soon, needs "
                  f"{args.min_age_min:.0f} min")
            continue
        if age > args.max_age_days * 1440:
            print(f"  {slug}: older than {args.max_age_days:.0f} days -- skipping")
            continue
        pending.append(slug)

    if not pending:
        print("Nothing pending.")
        return

    print(f"Pending verification: {', '.join(pending)}")
    if args.dry_run:
        return

    done = 0
    for slug in pending:
        if verify_run(args.site_dir, slug, args.nc_dir, args) is not None:
            done += 1

    idx["verified_updated"] = now.isoformat(timespec="seconds")
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)
    print(f"\nVerified {done} of {len(pending)} run(s). "
          f"Commit and push to publish.")


if __name__ == "__main__":
    main()
