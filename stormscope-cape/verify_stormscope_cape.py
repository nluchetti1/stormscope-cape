#!/usr/bin/env python3
"""
Verify a StormScope Cape Canaveral nowcast against observed MRMS and GLM.

Reads the NetCDF written by stormscope_cape.py, fetches observed MRMS
composite reflectivity and observed gridded GLM flash density at each
forecast valid time, regrids both to the same Cape window, and produces:

  *_refc.png  forecast vs observed composite reflectivity
  *_glm.png   forecast vs observed GLM flash density
  a contingency table for reflectivity, and flash-count totals for GLM

Retrospective (or at least already-elapsed) inits only -- the observations
have to exist.

Usage
-----
  python verify_stormscope_cape.py outputs/stormscope_cape_....nc
  python verify_stormscope_cape.py <file.nc> --zoom-km 80 --max-panels 6

Notes on the GLM comparison
---------------------------
GOESGLMGrid bins events into 5-minute windows, while StormScope steps at
10 minutes. --glm-bins controls how many consecutive 5-min observed bins
are summed to line up with one forecast step (default 2, i.e. the 10 min
ending at the valid time). I have not confirmed against NVIDIA's training
code whether their glm_density accumulates over the preceding 10 minutes
or a different window, so treat the absolute counts as approximate and the
spatial pattern as the meaningful comparison.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import plotstyle as ps

from scipy import ndimage

from earth2studio.data import MRMS, GOESGLMGrid, fetch_data
from earth2studio.utils.interp import NearestNeighborInterpolator


# ---------------------------------------------------------------------------
def great_circle_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2.0) ** 2)
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def zoom_slices(pad_dist_km, zoom_km):
    """Index slices for the sub-box within zoom_km of the pad.

    Operates on the stored distance-from-pad field, so it re-crops an
    existing output file without re-running the model.
    """
    inside = pad_dist_km <= zoom_km
    if not inside.any():
        raise SystemExit(f"--zoom-km {zoom_km} excludes every grid point")
    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def grid_spacing_km(lat, lon):
    """Median spacing between adjacent grid points, in km.

    Measured from the stored lat/lon rather than assumed, so this stays
    correct if the crop or the underlying grid ever changes.
    """
    dy = great_circle_km(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :])
    dx = great_circle_km(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    return float(np.median(np.concatenate([dy.ravel(), dx.ravel()])))


def circular_footprint(radius_px):
    """Boolean disc of the given radius in grid points."""
    n = int(np.ceil(radius_px))
    yy, xx = np.ogrid[-n:n + 1, -n:n + 1]
    return (xx * xx + yy * yy) <= radius_px * radius_px


def neighborhood_probability(field, thresh, dx_km, radius_km,
                             mode="nmep", smooth_km=None):
    """Neighborhood ensemble probability from a [member, y, x] field.

    mode="nmep"  Neighborhood Maximum Ensemble Probability. Each member's
                 binary exceedance field is dilated by a disc of radius
                 radius_km, then averaged across members. Reads as
                 "fraction of members with the event ANYWHERE within
                 radius_km of this point" -- the form that matches
                 standoff-distance thinking.

    mode="nep"   Neighborhood Ensemble Probability. Each member's binary
                 field is replaced by the FRACTION of points inside the
                 disc that exceed, then averaged. Smoother and less
                 aggressive than NMEP; closer to a fractions-skill view.

    smooth_km    Optional Gaussian smoothing applied after the ensemble
                 average. With a small ensemble NMEP is noisy, and light
                 smoothing (Schwartz & Sobash 2017) makes it far more
                 readable. Sigma is smooth_km / dx_km.

    Points invalid in every member come back NaN.
    """
    field = np.asarray(field)
    if field.ndim == 2:
        field = field[None]
    valid = np.isfinite(field).any(axis=0)

    # NaN must count as "no event", not propagate through the filter
    binary = (np.nan_to_num(field, nan=-np.inf) >= thresh).astype(np.float32)

    radius_px = radius_km / dx_km
    foot = circular_footprint(radius_px)

    per_member = np.empty_like(binary)
    if mode == "nmep":
        for m in range(binary.shape[0]):
            per_member[m] = ndimage.maximum_filter(
                binary[m], footprint=foot, mode="nearest")
    elif mode == "nep":
        kern = foot.astype(np.float32) / float(foot.sum())
        for m in range(binary.shape[0]):
            per_member[m] = ndimage.convolve(binary[m], kern, mode="nearest")
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'nmep' or 'nep'")

    p = per_member.mean(axis=0)
    if smooth_km:
        p = ndimage.gaussian_filter(p, sigma=smooth_km / dx_km, mode="nearest")
    return np.where(valid, p, np.nan)


def contingency(fcst, obs, thresh):
    """POD, FAR, CSI, bias over valid points. NaN where undefined."""
    m = np.isfinite(fcst) & np.isfinite(obs)
    f = fcst[m] >= thresh
    o = obs[m] >= thresh
    hits = np.sum(f & o)
    misses = np.sum(~f & o)
    fa = np.sum(f & ~o)
    pod = hits / (hits + misses) if (hits + misses) else np.nan
    far = fa / (hits + fa) if (hits + fa) else np.nan
    csi = hits / (hits + misses + fa) if (hits + misses + fa) else np.nan
    bias = (hits + fa) / (hits + misses) if (hits + misses) else np.nan
    return pod, far, csi, bias


def add_map_furniture(ax, proj):
    ps.add_furniture(ax, counties=True, gridlines=True, lw=0.85)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ncfile")
    ap.add_argument("--thresh", type=float, default=35.0,
                    help="Reflectivity threshold (dBZ) for contingency scores.")
    ap.add_argument("--zoom-km", type=float, default=None,
                    help="Re-crop plots to this radius (km) around the pad. "
                         "The file itself is unchanged. Try 60-120 for a "
                         "tight Cape view.")
    ap.add_argument("--glm-bins", type=int, default=2,
                    help="5-min observed GLM bins summed per 10-min forecast "
                         "step (2 = the 10 min ending at valid time).")
    ap.add_argument("--glm-thresh", type=float, default=None,
                    help="Flash-density threshold for the GLM hit/miss "
                         "summary. Default: the threshold stored in the file.")
    ap.add_argument("--nbhd-km", type=float, default=10.0,
                    help="Neighborhood radius (km) for the ensemble "
                         "probability panels. 0 disables and falls back to "
                         "point probabilities.")
    ap.add_argument("--nep-mode", choices=["nmep", "nep"], default="nmep",
                    help="nmep = P(event anywhere within radius); "
                         "nep = P from the fraction of points in the disc.")
    ap.add_argument("--nep-smooth-km", type=float, default=None,
                    help="Gaussian smoothing (km) applied after the ensemble "
                         "average. Useful with small ensembles; try ~half "
                         "the neighborhood radius.")
    ap.add_argument("--max-panels", type=int, default=6)
    ap.add_argument("--member", default="all",
                    help="Which ensemble member(s) to plot: an integer, a "
                         "comma list (0,2,4), or 'all' (default). With more "
                         "than one, panels are laid out one row per member "
                         "with observed on the bottom row.")
    ap.add_argument("--mean", action="store_true",
                    help="Add an ensemble-mean/max/probability summary figure "
                         "instead of per-member panels.")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ds = xr.open_dataset(args.ncfile)

    init = ds.attrs.get("init_time", "unknown")
    clat = ds.attrs.get("crop_center_lat", 28.585)
    clon = ds.attrs.get("crop_center_lon", -80.650)
    glm_thresh = (args.glm_thresh if args.glm_thresh is not None
                  else float(ds.attrs.get("glm_threshold", 1.0)))

    n_mem = int(ds.sizes.get("member", 1))
    if str(args.member).strip().lower() == "all":
        members = list(range(n_mem))
    else:
        members = [int(x) for x in str(args.member).split(",")]
        bad = [m for m in members if m < 0 or m >= n_mem]
        if bad:
            raise SystemExit(f"member(s) {bad} out of range; file has {n_mem}")

    if "pad_dist_km" in ds.coords:
        pad_d = ds["pad_dist_km"].values
    else:
        pad_d = great_circle_km(ds["lat"].values, ds["lon"].values, clat, clon)

    if args.zoom_km:
        zy, zx = zoom_slices(pad_d, args.zoom_km)
    else:
        zy = zx = slice(None)

    lat = ds["lat"].values[zy, zx]
    lon = ds["lon"].values[zy, zx]
    valid_times = ds["valid_time"].values

    print(f"file        : {args.ncfile}")
    print(f"init        : {init}")
    print(f"members     : {n_mem} in file, plotting {members}")
    print(f"plot window : {lat.shape[0]} x {lat.shape[1]} points"
          + (f"  (zoomed to {args.zoom_km:.0f} km)" if args.zoom_km else ""))
    print(f"glm thresh  : {glm_thresh}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proj = ccrs.PlateCarree()
    plt.rcParams.update({
        "figure.facecolor": "#0B1218", "savefig.facecolor": "#0B1218",
        "axes.facecolor": "#0B1218", "text.color": ps.INK,
        "axes.labelcolor": ps.INK, "font.family": "DejaVu Sans",
    })
    tgt_lat = torch.as_tensor(lat)
    tgt_lon = torch.as_tensor(lon % 360.0)

    # ---- observed MRMS -----------------------------------------------------
    mrms = MRMS()
    obs_refc, interp_r = [], None
    for t in valid_times:
        try:
            o, oc = fetch_data(mrms, time=[t], variable=np.array(["refc"]),
                               lead_time=np.array([np.timedelta64(0, "m")]),
                               device=device)
        except Exception as e:
            print(f"  no MRMS at {str(t)[:16]}: {type(e).__name__}")
            obs_refc.append(None)
            continue
        if interp_r is None:
            interp_r = NearestNeighborInterpolator(
                source_lats=oc["lat"], source_lons=oc["lon"],
                target_lats=tgt_lat, target_lons=tgt_lon,
                max_dist_km=12.0).to(device)
        obs_refc.append(interp_r(o).squeeze().detach().float().cpu().numpy())

    # ---- observed GLM ------------------------------------------------------
    glm_src = GOESGLMGrid(satellite="east")
    obs_glm, interp_g = [], None
    for t in valid_times:
        acc, oc = None, None
        for k in range(args.glm_bins):
            tb = t - np.timedelta64(5 * k, "m")
            try:
                o, oc = fetch_data(glm_src, time=[tb],
                                   variable=np.array(["glm_density"]),
                                   lead_time=np.array([np.timedelta64(0, "m")]),
                                   device=device)
            except Exception as e:
                print(f"  no GLM at {str(tb)[:16]}: {type(e).__name__}")
                continue
            acc = o if acc is None else acc + o
        if acc is None:
            obs_glm.append(None)
            continue
        if interp_g is None:
            interp_g = NearestNeighborInterpolator(
                source_lats=oc["lat"], source_lons=oc["lon"],
                target_lats=tgt_lat, target_lons=tgt_lon,
                max_dist_km=15.0).to(device)
        obs_glm.append(interp_g(acc).squeeze().detach().float().cpu().numpy())

    # ---- forecast fields ---------------------------------------------------
    refc_all = ds["mrms"].sel(mrms_channel="refc")
    glm_all = ds["mrms"].sel(mrms_channel="glm_density")

    def fc(da, m, k):
        """Member m, valid-time index k, cropped to the plot window."""
        return da.isel(member=m).sel(valid_time=valid_times[k]).values[zy, zx]

    t0 = np.datetime64(init.replace("+00:00", "")) if init != "unknown" \
        else valid_times[0] - np.timedelta64(10, "m")

    # ---- tables ------------------------------------------------------------
    print(f"\nReflectivity vs observed MRMS, threshold {args.thresh:.0f} dBZ")
    print("mem  lead   valid_time         POD    FAR    CSI   bias   "
          "max_f   max_o")
    print("-" * 76)
    csi_by_lead = {k: [] for k in range(len(valid_times))}
    for m in members:
        for k, t in enumerate(valid_times):
            f = fc(refc_all, m, k)
            lead = int((t - t0) / np.timedelta64(1, "m"))
            if obs_refc[k] is None:
                print(f"{m:3d}  {lead:4d}m  {str(t)[:16]}    --     --     "
                      f"--     --  {np.nanmax(f):6.1f}      --")
                continue
            o = obs_refc[k]
            pod, far, csi, bias = contingency(f, o, args.thresh)
            csi_by_lead[k].append(csi)
            print(f"{m:3d}  {lead:4d}m  {str(t)[:16]}  {pod:5.2f}  {far:5.2f}  "
                  f"{csi:5.2f}  {bias:5.2f}  {np.nanmax(f):6.1f}  "
                  f"{np.nanmax(o):6.1f}")
        if len(members) > 1:
            print("-" * 76)

    if len(members) > 1:
        print("\nCSI spread across members (min / mean / max)")
        for k, t in enumerate(valid_times):
            vals = [c for c in csi_by_lead[k] if np.isfinite(c)]
            lead = int((t - t0) / np.timedelta64(1, "m"))
            if not vals:
                print(f"  {lead:4d}m  --")
                continue
            print(f"  {lead:4d}m  {min(vals):.2f} / {np.mean(vals):.2f} / "
                  f"{max(vals):.2f}")

    print(f"\nGLM flash density vs observed, threshold {glm_thresh}")
    print("lead   valid_time      fcst_sum   obs_sum  fcst_cells  obs_cells  "
          "POD    FAR")
    print("-" * 82)
    for k, t in enumerate(valid_times):
        lead = int((t - t0) / np.timedelta64(1, "m"))
        sums = [float(np.nansum(fc(glm_all, m, k))) for m in members]
        cells = [int(np.nansum(fc(glm_all, m, k) > glm_thresh)) for m in members]
        fs, fcell = float(np.mean(sums)), int(np.mean(cells))
        if obs_glm[k] is None:
            print(f"{lead:4d}m  {str(t)[:16]}  {fs:9.1f}        --  "
                  f"{fcell:10d}         --     --     --")
            continue
        o = obs_glm[k]
        pods, fars = [], []
        for m in members:
            p, fr, _, _ = contingency(fc(glm_all, m, k), o, glm_thresh)
            pods.append(p)
            fars.append(fr)
        pm = np.nanmean(pods) if np.any(np.isfinite(pods)) else np.nan
        fm = np.nanmean(fars) if np.any(np.isfinite(fars)) else np.nan
        print(f"{lead:4d}m  {str(t)[:16]}  {fs:9.1f}  {float(np.nansum(o)):8.1f}  "
              f"{fcell:10d}  {int(np.nansum(o > glm_thresh)):9d}  "
              f"{pm:5.2f}  {fm:5.2f}")
    if len(members) > 1:
        print("  (fcst columns are the ensemble mean across "
              f"{len(members)} members)")

    # ---- panel plots -------------------------------------------------------
    idx = np.unique(np.linspace(0, len(valid_times) - 1,
                                min(args.max_panels, len(valid_times))
                                ).astype(int))
    n = len(idx)

    def panel_fig(get_f, get_o, title, cbar_label, **kw):
        """One row per plotted member, plus a bottom row of observations."""
        levels, fmt = kw.pop("_levels", None), kw.pop("_fmt", None)
        kw["_levels"], kw["_fmt"] = levels, fmt   # kept for the colorbar
        draw = {k2: v2 for k2, v2 in kw.items() if not k2.startswith("_")}
        kw = dict(kw); kw.update(draw)
        nrow = len(members) + 1
        fig, axes = plt.subplots(nrow, n, figsize=(3.1 * n, 3.0 * nrow),
                                 subplot_kw={"projection": proj},
                                 squeeze=False)
        im = None
        for r, m in enumerate(members):
            for col, k in enumerate(idx):
                lead = int((valid_times[k] - t0) / np.timedelta64(1, "m"))
                ax = axes[r, col]
                add_map_furniture(ax, proj)
                im = ax.pcolormesh(lon, lat, get_f(m, k), transform=proj,
                                   shading="auto", zorder=3, **draw)
                ax.set_title(f"MEMBER {m}   F+{lead:03d}m", fontsize=8.5,
                             color=ps.INK, fontweight="bold", pad=3)
                ax.text(0.5, -0.035, ps.zulu(valid_times[k]),
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=7.2, color=ps.SIGNAL,
                        family="DejaVu Sans Mono")
        for col, k in enumerate(idx):
            lead = int((valid_times[k] - t0) / np.timedelta64(1, "m"))
            ax = axes[-1, col]
            add_map_furniture(ax, proj)
            o = get_o(k)
            if o is not None:
                ax.pcolormesh(lon, lat, o, transform=proj, shading="auto",
                              zorder=3, **draw)
                ax.set_title(f"OBSERVED   F+{lead:03d}m", fontsize=8.5,
                             color=ps.CYAN, fontweight="bold", pad=3)
            else:
                ax.set_title(f"obs missing   F+{lead:03d}m", fontsize=8.5,
                             color=ps.FAINT, pad=3)
            ax.text(0.5, -0.035, ps.zulu(valid_times[k]),
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7.2, color=ps.SIGNAL,
                    family="DejaVu Sans Mono")
        cb = fig.colorbar(im, ax=axes, orientation="horizontal",
                          pad=0.03, shrink=0.35)
        ps.style_colorbar(cb, cbar_label, levels=kw.get("_levels"),
                          fmt=kw.get("_fmt"))
        fig.suptitle(title, fontsize=12.5, color=ps.INK, fontweight="bold")
        return fig

    base = os.path.splitext(os.path.basename(args.ncfile))[0]
    zt = f"   ZOOM {args.zoom_km:.0f} km" if args.zoom_km else ""
    _dbz_cmap, _dbz_norm = ps.dbz_cmap_norm()

    fig = panel_fig(
        lambda m, k: np.where(fc(refc_all, m, k) <= 0, np.nan,
                              fc(refc_all, m, k)),
        lambda k: (None if obs_refc[k] is None
                   else np.where(obs_refc[k] <= 0, np.nan, obs_refc[k])),
        f"StormScope Cape Canaveral vs MRMS   |   INIT {ps.zulu(init[:16])}{zt}",
        "Composite reflectivity [dBZ]",
        cmap=_dbz_cmap, norm=_dbz_norm,
        _levels=ps.NWS_DBZ_LEVELS, _fmt=lambda v: f"{int(v)}")
    out_r = os.path.join(args.outdir, f"{base}_refc.png")
    fig.savefig(out_r, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_r}")

    # shared log scale so forecast and observed are directly comparable
    fmax = float(np.nanmax(glm_all.values[..., zy, zx]))
    omax = max((float(np.nanmax(o)) for o in obs_glm if o is not None),
               default=1.0)
    vmax = max(fmax, omax, 2.0)
    _glm_cmap, norm = ps.glm_cmap_norm(vmax, glm_thresh)

    fig = panel_fig(
        lambda m, k: np.where(fc(glm_all, m, k) <= glm_thresh, np.nan,
                              fc(glm_all, m, k)),
        lambda k: (None if obs_glm[k] is None
                   else np.where(obs_glm[k] <= glm_thresh, np.nan, obs_glm[k])),
        f"StormScope GLM flash density vs observed   |   "
        f"INIT {ps.zulu(init[:16])}{zt}",
        f"GLM flash density [events/cell, > {glm_thresh}]",
        cmap=_glm_cmap, norm=norm)
    out_g = os.path.join(args.outdir, f"{base}_glm.png")
    fig.savefig(out_g, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_g}")

    # ---- ensemble lightning probability, if there is an ensemble -----------
    if n_mem > 1 and "p_lightning" in ds:
        fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.6),
                                 subplot_kw={"projection": proj})
        axes = np.atleast_1d(axes)
        im = None
        for col, k in enumerate(idx):
            t = valid_times[k]
            lead = int((t - t0) / np.timedelta64(1, "m"))
            p = ds["p_lightning"].sel(valid_time=t).values[zy, zx]
            im = axes[col].pcolormesh(lon, lat, np.where(p <= 0, np.nan, p),
                                      transform=proj, cmap="plasma",
                                      vmin=0.0, vmax=1.0, shading="auto")
            add_map_furniture(axes[col], proj)
            axes[col].set_title(f"F+{lead}m P(lightning)", fontsize=9)
        fig.colorbar(im, ax=axes, orientation="horizontal",
                     label="Ensemble fraction", pad=0.05, shrink=0.4)
        out_p = os.path.join(args.outdir, f"{base}_plight.png")
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_p}")

        # ---- ensemble summary, with neighborhood probabilities ----------
        dx_km = grid_spacing_km(lat, lon)
        _p_cmap, _p_norm = ps.prob_cmap_norm()
        use_nbhd = args.nbhd_km and args.nbhd_km > 0
        if use_nbhd:
            rpx = args.nbhd_km / dx_km
            npts = int(circular_footprint(rpx).sum())
            tag = args.nep_mode.upper()
            plab = f"{tag} {args.nbhd_km:.0f} km"
            print(f"\nNeighborhood probability: {tag}, radius "
                  f"{args.nbhd_km:.0f} km on a {dx_km:.2f} km grid "
                  f"= {rpx:.2f} px ({npts} points in disc)"
                  + (f", smoothed {args.nep_smooth_km:.0f} km"
                     if args.nep_smooth_km else ""))
            print(f"  {n_mem} members x {npts} points -> up to "
                  f"{n_mem * npts} distinct probability levels "
                  f"(point probability would give {n_mem + 1})")
        else:
            plab = "point"

        def prob_field(channel, thresh, k):
            """Neighborhood (or point) probability at valid-time index k."""
            f = (ds["mrms"].sel(mrms_channel=channel)
                 .isel(valid_time=k).values[..., zy, zx])
            if not use_nbhd:
                with np.errstate(invalid="ignore"):
                    return np.nanmean((f >= thresh).astype("float32"), axis=0)
            return neighborhood_probability(
                f, thresh, dx_km, args.nbhd_km,
                mode=args.nep_mode, smooth_km=args.nep_smooth_km)

        rows = [
            ("refc_ens_mean", "Ens mean refc [dBZ]", None, None,
             dict(cmap=_dbz_cmap, norm=_dbz_norm)),
            ("refc_ens_max", "Ens max refc [dBZ]", None, None,
             dict(cmap=_dbz_cmap, norm=_dbz_norm)),
            (None, f"{plab} P(refc>={int(args.thresh)}dBZ)",
             "refc", args.thresh,
             dict(cmap=_p_cmap, norm=_p_norm)),
            (None, f"{plab} P(lightning)", "glm_density", glm_thresh,
             dict(cmap=_p_cmap, norm=_p_norm)),
        ]
        rows = [r for r in rows if r[0] is None or r[0] in ds]

        fig, axes = plt.subplots(len(rows), n, figsize=(3.1 * n, 3.0 * len(rows)),
                                 subplot_kw={"projection": proj}, squeeze=False)
        for r, (var, label, chan, thr, kw) in enumerate(rows):
            im = None
            for col, k in enumerate(idx):
                lead = int((valid_times[k] - t0) / np.timedelta64(1, "m"))
                if var is not None:
                    d = ds[var].sel(valid_time=valid_times[k]).values[zy, zx]
                else:
                    d = prob_field(chan, thr, k)
                add_map_furniture(axes[r, col], proj)
                im = axes[r, col].pcolormesh(lon, lat,
                                             np.where(d <= 0, np.nan, d),
                                             transform=proj, shading="auto",
                                             zorder=3, **kw)
                axes[r, col].set_title(f"{label}   F+{lead:03d}m", fontsize=8,
                                       color=ps.INK, pad=3)
                axes[r, col].text(0.5, -0.03, ps.zulu(valid_times[k]),
                                  transform=axes[r, col].transAxes,
                                  ha="center", va="top", fontsize=6.6,
                                  color=ps.SIGNAL, family="DejaVu Sans Mono")
            cb = fig.colorbar(im, ax=axes[r, :].tolist(),
                              orientation="vertical", shrink=0.85, pad=0.01)
            ps.style_colorbar(cb, "")
        fig.suptitle(f"Ensemble summary   |   {n_mem} members   |   {plab}"
                     f"   |   INIT {ps.zulu(init[:16])}{zt}",
                     fontsize=12.5, color=ps.INK, fontweight="bold")

        # ---- pad-ring neighborhood probability time series --------------
        print(f"\n{plab} probability at the pad")
        print("lead   valid_time        P(lgt)  P(refc)")
        print("-" * 46)
        pd_z = pad_d[zy, zx]
        pad_px = np.unravel_index(np.argmin(pd_z), pd_z.shape)
        for k, t in enumerate(valid_times):
            lead = int((t - t0) / np.timedelta64(1, "m"))
            pl = prob_field("glm_density", glm_thresh, k)[pad_px]
            pr = prob_field("refc", args.thresh, k)[pad_px]
            print(f"{lead:4d}m  {str(t)[:16]}   {pl:5.2f}   {pr:5.2f}")
        suffix = (f"_{args.nep_mode}{int(args.nbhd_km)}km"
                  if use_nbhd else "_point")
        out_s = os.path.join(args.outdir, f"{base}_ensemble{suffix}.png")
        fig.savefig(out_s, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_s}")


if __name__ == "__main__":
    main()
