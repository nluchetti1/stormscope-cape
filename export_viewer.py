#!/usr/bin/env python3
"""
Turn StormScope NetCDF output into a static PNG frame set for the web viewer.

CPU only -- no GPU, no earth2studio needed unless you pass --with-obs.
Run it on the pod, or locally after pulling the .nc files down.

Produces, under --site-dir:

    runs/index.json                     list of all runs, newest first
    runs/<INIT>/manifest.json           variables, leads, colorbars
    runs/<INIT>/<var>/f<NNN>.png        one frame per lead time
    runs/<INIT>/<var>/colorbar.png      horizontal scale for that variable

Frames are transparent PNGs with light map furniture, sized identically
across every variable so the viewer can swap them without reflow.

Usage
-----
  python export_viewer.py outputs/stormscope_cape_*_ens08.nc

  python export_viewer.py outputs/*.nc --zoom-km 100 --nbhd-km 10 \
      --members 0,1,2 --with-obs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import ndimage

import plotstyle as ps

FIG_IN = 7.6          # inches; every frame is square and identical
DPI = 165


# ---------------------------------------------------------------------------
# neighborhood probability (duplicated from the verify script on purpose, so
# this exporter needs only numpy/scipy/xarray/matplotlib/cartopy)
# ---------------------------------------------------------------------------
def great_circle_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2.0) ** 2)
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def grid_spacing_km(lat, lon):
    dy = great_circle_km(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :])
    dx = great_circle_km(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    return float(np.median(np.concatenate([dy.ravel(), dx.ravel()])))


def circular_footprint(radius_px):
    n = int(np.ceil(radius_px))
    yy, xx = np.ogrid[-n:n + 1, -n:n + 1]
    return (xx * xx + yy * yy) <= radius_px * radius_px


def neighborhood_probability(field, thresh, dx_km, radius_km,
                             mode="nmep", smooth_km=None):
    """See verify_stormscope_cape.py for the full explanation."""
    field = np.asarray(field)
    if field.ndim == 2:
        field = field[None]
    valid = np.isfinite(field).any(axis=0)
    binary = (np.nan_to_num(field, nan=-np.inf) >= thresh).astype(np.float32)
    foot = circular_footprint(radius_km / dx_km)
    out = np.empty_like(binary)
    if mode == "nmep":
        for m in range(binary.shape[0]):
            out[m] = ndimage.maximum_filter(binary[m], footprint=foot,
                                            mode="nearest")
    else:
        kern = foot.astype(np.float32) / float(foot.sum())
        for m in range(binary.shape[0]):
            out[m] = ndimage.convolve(binary[m], kern, mode="nearest")
    p = out.mean(axis=0)
    if smooth_km:
        p = ndimage.gaussian_filter(p, sigma=smooth_km / dx_km, mode="nearest")
    return np.where(valid, p, np.nan)


def zoom_slices(pad_dist_km, zoom_km):
    inside = pad_dist_km <= zoom_km
    if not inside.any():
        raise SystemExit(f"--zoom-km {zoom_km} excludes every grid point")
    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render_frame(path, lat, lon, data, cmap, norm, mask_at, *,
                 label="", valid="", init=None, lead=None, members=None,
                 extra=None, rings=None, center=None):
    """One square frame: shaded field, map furniture, burnt-in Zulu stamps."""
    fig = plt.figure(figsize=(FIG_IN, FIG_IN), facecolor="#0B1218")
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()],
                  crs=ccrs.PlateCarree())
    ax.set_facecolor("#0B1218")

    ps.add_furniture(ax, rings=rings, center=center)

    d = np.where(np.isfinite(data) & (data > mask_at), data, np.nan)
    ax.pcolormesh(lon, lat, d, transform=ccrs.PlateCarree(),
                  cmap=cmap, norm=norm, shading="auto", zorder=3)

    ps.annotate(ax, label, valid, init=init, lead=lead, members=members,
                extra=extra)
    ax.set_frame_on(False)
    fig.savefig(path, dpi=DPI, facecolor="#0B1218", pad_inches=0)
    plt.close(fig)


def render_colorbar(path, cmap, norm, label, levels=None, fmt=None):
    fig = plt.figure(figsize=(7.4, 0.92), facecolor="#0B1218")
    ax = fig.add_axes([0.035, 0.46, 0.93, 0.30])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax, orientation="horizontal")
    ps.style_colorbar(cb, label, levels=levels, fmt=fmt)
    fig.savefig(path, dpi=DPI, facecolor="#0B1218", bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
def build_variables(ds, args, zy, zx, lat, lon, dx_km, glm_thresh, members):
    """Assemble the ordered list of variables to export.

    Each entry: (id, label, units, cmap, norm, mask_at, getter(k) -> 2D array)
    """
    refc_all = ds["mrms"].sel(mrms_channel="refc")
    glm_all = ds["mrms"].sel(mrms_channel="glm_density")
    n_mem = int(ds.sizes.get("member", 1))

    dbz_cmap, dbz_norm = ps.dbz_cmap_norm()
    p_cmap, p_norm = ps.prob_cmap_norm()
    glm_max = float(np.nanmax(glm_all.values[..., zy, zx]))
    glm_cmap, glm_norm = ps.glm_cmap_norm(glm_max, glm_thresh)

    def crop(da, m, k):
        return da.isel(member=m, valid_time=k).values[zy, zx]

    def stack(da, k):
        return da.isel(valid_time=k).values[..., zy, zx]

    out = []
    # (id, label, units, cmap, norm, mask_at, getter, group, group_label,
    #  variant, variant_label)
    if n_mem > 1:
        out.append(("refc_ens_mean", "Ensemble mean reflectivity", "dBZ",
                    dbz_cmap, dbz_norm, 4.9,
                    lambda k: ds["refc_ens_mean"].isel(valid_time=k).values[zy, zx],
                    "refc", "Reflectivity", "mean", "Mean"))
        out.append(("refc_ens_max", "Ensemble max reflectivity", "dBZ",
                    dbz_cmap, dbz_norm, 4.9,
                    lambda k: ds["refc_ens_max"].isel(valid_time=k).values[zy, zx],
                    "refc", "Reflectivity", "max", "Max"))

        nb = args.nbhd_km
        tag = args.nep_mode.upper()
        if nb and nb > 0:
            out.append((
                "p_refc", f"{tag} {nb:.0f} km  P(refc \u2265 {args.thresh:.0f} dBZ)",
                "probability", p_cmap, p_norm, 0.049,
                lambda k: neighborhood_probability(
                    stack(refc_all, k), args.thresh, dx_km, nb,
                    mode=args.nep_mode, smooth_km=args.nep_smooth_km),
                "p_refc", f"P(refc \u2265 {args.thresh:.0f} dBZ)", "prob", "Prob"))
            out.append((
                "p_lightning", f"{tag} {nb:.0f} km  P(lightning)",
                "probability", p_cmap, p_norm, 0.049,
                lambda k: neighborhood_probability(
                    stack(glm_all, k), glm_thresh, dx_km, nb,
                    mode=args.nep_mode, smooth_km=args.nep_smooth_km),
                "p_lightning", "P(lightning)", "prob", "Prob"))
        else:
            out.append(("p_refc", f"P(refc \u2265 {args.thresh:.0f} dBZ)",
                        "probability", p_cmap, p_norm, 0.049,
                        lambda k: np.nanmean(
                            (stack(refc_all, k) >= args.thresh).astype("f4"), 0),
                        "p_refc", f"P(refc \u2265 {args.thresh:.0f} dBZ)",
                        "prob", "Prob"))
            out.append(("p_lightning", "P(lightning)", "probability",
                        p_cmap, p_norm, 0.049,
                        lambda k: np.nanmean(
                            (stack(glm_all, k) > glm_thresh).astype("f4"), 0),
                        "p_lightning", "P(lightning)", "prob", "Prob"))

        out.append(("glm_ens_mean", "Ensemble mean GLM flash density",
                    "events/cell", glm_cmap, glm_norm, glm_thresh,
                    lambda k: np.nanmean(stack(glm_all, k), axis=0),
                    "glm", "GLM flash density", "mean", "Mean"))

    for m in members:
        out.append((f"refc_m{m}", f"Member {m} reflectivity", "dBZ",
                    dbz_cmap, dbz_norm, 4.9,
                    lambda k, m=m: crop(refc_all, m, k),
                    "refc", "Reflectivity", f"m{m}", f"m{m}"))
        out.append((f"glm_m{m}", f"Member {m} GLM flash density",
                    "events/cell", glm_cmap, glm_norm, glm_thresh,
                    lambda k, m=m: crop(glm_all, m, k),
                    "glm", "GLM flash density", f"m{m}", f"m{m}"))

    return out


def fetch_observed(ds, valid_times, lat, lon):
    """Observed MRMS reflectivity on the plot grid. Needs earth2studio."""
    try:
        import torch
        from earth2studio.data import MRMS, fetch_data
        from earth2studio.utils.interp import NearestNeighborInterpolator
    except ImportError as e:
        print(f"  --with-obs needs earth2studio in this env ({e}); skipping")
        return None

    dev = torch.device("cpu")
    mrms = MRMS()
    interp = None
    frames = []
    for t in valid_times:
        try:
            o, oc = fetch_data(mrms, time=[t], variable=np.array(["refc"]),
                               lead_time=np.array([np.timedelta64(0, "m")]),
                               device=dev)
        except Exception as e:
            print(f"  no observed MRMS at {str(t)[:16]}: {type(e).__name__}")
            frames.append(None)
            continue
        if interp is None:
            interp = NearestNeighborInterpolator(
                source_lats=oc["lat"], source_lons=oc["lon"],
                target_lats=torch.as_tensor(lat),
                target_lons=torch.as_tensor(lon % 360.0),
                max_dist_km=12.0)
        frames.append(interp(o).squeeze().detach().float().cpu().numpy())
    return frames


# ---------------------------------------------------------------------------
def export_run(ncfile, args):
    ds = xr.open_dataset(ncfile)
    init_iso = ds.attrs.get("init_time", "unknown")
    init_dt = datetime.fromisoformat(init_iso) if init_iso != "unknown" else None
    slug = init_dt.strftime("%Y%m%dT%H%MZ") if init_dt else \
        os.path.splitext(os.path.basename(ncfile))[0]

    clat = ds.attrs.get("crop_center_lat", 28.585)
    clon = ds.attrs.get("crop_center_lon", -80.650)
    glm_thresh = float(ds.attrs.get("glm_threshold", 1.0))
    n_mem = int(ds.sizes.get("member", 1))

    pad_d = (ds["pad_dist_km"].values if "pad_dist_km" in ds.coords
             else great_circle_km(ds["lat"].values, ds["lon"].values, clat, clon))
    zy, zx = zoom_slices(pad_d, args.zoom_km) if args.zoom_km else (slice(None),) * 2
    lat = ds["lat"].values[zy, zx]
    lon = ds["lon"].values[zy, zx]
    dx_km = grid_spacing_km(lat, lon)

    if str(args.members).strip().lower() == "all":
        members = list(range(n_mem))
    elif str(args.members).strip() == "":
        members = []
    else:
        members = [int(x) for x in str(args.members).split(",")
                   if 0 <= int(x) < n_mem]

    valid_times = ds["valid_time"].values
    t0 = np.datetime64(init_iso.replace("+00:00", "")) if init_dt else \
        valid_times[0] - np.timedelta64(10, "m")

    run_dir = os.path.join(args.site_dir, "runs", slug)
    if os.path.isdir(run_dir) and args.clean:
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{os.path.basename(ncfile)}")
    print(f"  init {init_iso} | {n_mem} member(s) | "
          f"{lat.shape[0]}x{lat.shape[1]} pts @ {dx_km:.2f} km")

    variables = build_variables(ds, args, zy, zx, lat, lon, dx_km,
                                glm_thresh, members)

    obs = fetch_observed(ds, valid_times, lat, lon) if args.with_obs else None
    if obs is not None:
        variables.append(("obs_refc", "Observed reflectivity (MRMS)", "dBZ",
                          *ps.dbz_cmap_norm(), 4.9,
                          lambda k: obs[k] if obs[k] is not None
                          else np.full(lat.shape, np.nan),
                          "refc", "Reflectivity", "obs", "Obs"))

    leads = [{"minutes": int((t - t0) / np.timedelta64(1, "m")),
              "valid": ps.zulu(t)}
             for t in valid_times]

    init_z = ps.zulu(init_iso[:16]) if init_dt else "—"
    run_done = ds.attrs.get("run_completed", "")
    run_z = ps.zulu(run_done[:16]) if run_done else ""
    run_kind = ds.attrs.get("run_kind",
                            "ensemble" if n_mem > 1 else "deterministic")
    rings = [float(r) for r in args.rings.split(",")] if args.rings else None
    center = (float(clat), float(clon))
    dom = (f"RUN {run_z}" if run_z else f"{lat.shape[1]}x{lat.shape[0]}")

    var_meta = []
    for (vid, label, units, cmap, norm, mask_at, getter,
         group, group_label, variant, variant_label) in variables:
        vdir = os.path.join(run_dir, vid)
        os.makedirs(vdir, exist_ok=True)
        if units == "dBZ":
            levels, fmt = ps.NWS_DBZ_LEVELS, lambda v: f"{int(v)}"
        elif units == "probability":
            levels, fmt = ps.PROB_LEVELS, lambda v: f"{int(round(v*100))}"
        else:
            levels = fmt = None

        files = []
        for k, ld in enumerate(leads):
            fn = f"f{ld['minutes']:03d}.png"
            render_frame(os.path.join(vdir, fn), lat, lon, getter(k),
                         cmap, norm, mask_at,
                         label=label, valid=ld["valid"], init=init_z,
                         lead=ld["minutes"], members=n_mem, extra=dom,
                         rings=rings, center=center)
            files.append(f"{vid}/{fn}")
        cb_label = (f"{label} [%]" if units == "probability"
                    else f"{label} [{units}]")
        render_colorbar(os.path.join(vdir, "colorbar.png"), cmap, norm,
                        cb_label, levels=levels, fmt=fmt)
        var_meta.append({"id": vid, "label": label, "units": units,
                         "group": group, "group_label": group_label,
                         "variant": variant, "variant_label": variant_label,
                         "colorbar": f"{vid}/colorbar.png", "frames": files})
        print(f"    {vid:<16} {len(files)} frames")

    manifest = {
        "init": init_iso,
        "slug": slug,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "members": n_mem,
        "run_completed": run_done,
        "run_kind": run_kind,
        "runtime_sec": ds.attrs.get("runtime_sec"),
        "model": ds.attrs.get("model", "NVIDIA StormScope"),
        "satellite": ds.attrs.get("satellite", ""),
        "domain": {
            "center_lat": float(clat), "center_lon": float(clon),
            "lat_min": float(lat.min()), "lat_max": float(lat.max()),
            "lon_min": float(lon.min()), "lon_max": float(lon.max()),
            "grid_km": round(dx_km, 2),
            "nx": int(lat.shape[1]), "ny": int(lat.shape[0]),
        },
        "thresholds": {"refc_dbz": args.thresh, "glm": glm_thresh},
        "neighborhood": ({"radius_km": args.nbhd_km, "mode": args.nep_mode,
                          "smooth_km": args.nep_smooth_km}
                         if args.nbhd_km else None),
        "note": ds.attrs.get(
            "note", "Research nowcast guidance. Not an LLCC evaluation; "
                    "encodes no launch commit criteria."),
        "leads": leads,
        "variables": var_meta,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return {"slug": slug, "init": init_iso, "members": n_mem,
            "run_completed": run_done, "run_kind": run_kind,
            "leads": len(leads), "generated": manifest["generated"]}


def rebuild_index(site_dir, keep):
    """Rewrite runs/index.json from whatever run directories exist."""
    runs_dir = os.path.join(site_dir, "runs")
    entries = []
    for slug in sorted(os.listdir(runs_dir)):
        mpath = os.path.join(runs_dir, slug, "manifest.json")
        if not os.path.isfile(mpath):
            continue
        with open(mpath) as f:
            m = json.load(f)
        entries.append({"slug": m["slug"], "init": m["init"],
                        "members": m["members"],
                        "run_completed": m.get("run_completed", ""),
                        "run_kind": m.get("run_kind",
                                          "ensemble" if m["members"] > 1
                                          else "deterministic"),
                        "leads": len(m["leads"]),
                        "generated": m["generated"]})

    entries.sort(key=lambda e: e["init"], reverse=True)

    if keep and len(entries) > keep:
        for e in entries[keep:]:
            path = os.path.join(runs_dir, e["slug"])
            print(f"  pruning old run {e['slug']}")
            shutil.rmtree(path, ignore_errors=True)
        entries = entries[:keep]

    with open(os.path.join(runs_dir, "index.json"), "w") as f:
        json.dump({"runs": entries,
                   "updated": datetime.now(timezone.utc).isoformat(
                       timespec="seconds")}, f, indent=2)
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ncfiles", nargs="+")
    ap.add_argument("--site-dir", default="site",
                    help="Root of the static site (default: site).")
    ap.add_argument("--zoom-km", type=float, default=120.0,
                    help="Crop radius around the pad, km. 0 = full window.")
    ap.add_argument("--thresh", type=float, default=35.0,
                    help="Reflectivity threshold (dBZ) for probabilities.")
    ap.add_argument("--nbhd-km", type=float, default=10.0,
                    help="Neighborhood radius (km). 0 = point probability.")
    ap.add_argument("--nep-mode", choices=["nmep", "nep"], default="nmep")
    ap.add_argument("--nep-smooth-km", type=float, default=None)
    ap.add_argument("--members", default="all",
                    help="Per-member frames to export: 'all', '0,1,2', or "
                         "'' for none (default). Each member roughly doubles "
                         "the frame count.")
    ap.add_argument("--rings", default="",
                    help="Comma-separated range rings in nmi drawn around "
                         "the pad, e.g. '5,10'. Empty (default) draws none.")
    ap.add_argument("--with-obs", action="store_true",
                    help="Also export observed MRMS. Needs earth2studio and "
                         "network access; only works for elapsed valid times.")
    ap.add_argument("--keep", type=int, default=10,
                    help="Keep only the N newest runs; older ones are "
                         "deleted so the repo does not grow without bound.")
    ap.add_argument("--clean", action="store_true",
                    help="Delete an existing directory for the same init "
                         "before re-exporting it.")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.site_dir, "runs"), exist_ok=True)

    for nc in args.ncfiles:
        export_run(nc, args)

    entries = rebuild_index(args.site_dir, args.keep)

    total = 0
    for root, _, files in os.walk(os.path.join(args.site_dir, "runs")):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    print(f"\n{len(entries)} run(s) in {args.site_dir}/runs, "
          f"{total / 1e6:.1f} MB total")
    print("Next: commit and push, or run ./deploy_pages.sh")


if __name__ == "__main__":
    main()
