#!/usr/bin/env python3
"""
Shared plotting style for StormScope Cape Canaveral products.

Everything that draws a map imports from here, so the web frames and the
verification panels use identical palettes, furniture and annotation.

Design notes
------------
Reflectivity uses the standard NWS Level-3 dBZ palette on 5 dBZ steps
rather than a continuous perceptual map. Continuous maps like turbo look
attractive but destroy the reading a forecaster actually does: with the
NWS palette you know 40 dBZ is orange without consulting a colorbar, and
the discrete steps make gradients legible as gradients. Probability and
flash density are the opposite case -- no community-standard palette
exists, so they use perceptual ramps with explicit discrete levels.

Frames carry their own valid time in Zulu. The viewer shows it too, but
burning it in means a screenshot pulled into a briefing is unambiguous
about what it is and when it was valid.
"""

from __future__ import annotations

import os

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------
INK = "#E8EFF5"
DIM = "#93A6B5"
FAINT = "#5C6D7A"
COAST = "#B8C9D6"
STATE = "#7E909C"
COUNTY = "#46555F"
SIGNAL = "#FF9E1B"
CYAN = "#4FC3D9"

# NWS Level-3 composite reflectivity, 5 dBZ steps from 5 to 75.
NWS_DBZ_COLORS = [
    "#04E9E7", "#019FF4", "#0300F4", "#02FD02", "#01C501", "#008E00",
    "#FDF802", "#E5BC00", "#FD9500", "#FD0000", "#D40000", "#BC0000",
    "#F800FD", "#9854C6",
]
NWS_DBZ_LEVELS = list(range(5, 76, 5))


def dbz_cmap_norm():
    """NWS reflectivity colormap and its boundary norm."""
    cmap = mcolors.ListedColormap(NWS_DBZ_COLORS, name="nws_dbz")
    cmap.set_under(alpha=0.0)
    cmap.set_over(NWS_DBZ_COLORS[-1])
    cmap.set_bad(alpha=0.0)
    # 15 boundaries -> 14 intervals, matching the 14 palette colours.
    # Values above 75 dBZ fall to set_over; nothing real reaches it.
    return cmap, mcolors.BoundaryNorm(NWS_DBZ_LEVELS, cmap.N)


PROB_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
               0.60, 0.70, 0.80, 0.90, 1.00]
PROB_COLORS = [
    "#1B3A63", "#20558C", "#1E77A8", "#22A0A8", "#38B87E",
    "#8DC63F", "#E8D02E", "#F2A32B", "#EE6C24", "#DB2B27",
]


def prob_cmap_norm():
    """Discrete probability palette, cool low to warm high."""
    cmap = mcolors.ListedColormap(PROB_COLORS, name="prob")
    cmap.set_under(alpha=0.0)
    cmap.set_bad(alpha=0.0)
    return cmap, mcolors.BoundaryNorm(PROB_LEVELS, cmap.N)


def glm_cmap_norm(vmax, thresh=1.0):
    """Flash density: dark violet through ember to white-hot, log scaled."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "glm", ["#2A1B4D", "#6B2B8C", "#B33A7A", "#E86A3C",
                "#F5B32E", "#FFF0C4"])
    cmap.set_under(alpha=0.0)
    cmap.set_bad(alpha=0.0)
    return cmap, mcolors.LogNorm(vmin=max(thresh, 0.5), vmax=max(vmax, 2.0))


# ---------------------------------------------------------------------------
# map furniture
# ---------------------------------------------------------------------------
_SCALE = None      # resolved once and cached
_COUNTIES = None

NMI_KM = 1.852


def _counties(scale):
    global _COUNTIES
    if _COUNTIES is None:
        _COUNTIES = cfeature.NaturalEarthFeature(
            "cultural", "admin_2_counties", scale,
            facecolor="none", edgecolor=COUNTY)
    return _COUNTIES


def add_furniture(ax, counties=True, gridlines=True, rings=None,
                  center=None, lw=1.0):
    """Coastline, state and county lines, optional graticule and range rings.

    Cartopy fetches Natural Earth shapefiles on first use. On a restricted
    network that fails, so this steps 10m -> 50m -> 110m -> nothing rather
    than aborting the render. SKIP_MAP_FURNITURE=1 bypasses it entirely.
    """
    global _SCALE
    if os.environ.get("SKIP_MAP_FURNITURE"):
        return

    for sc in ([_SCALE] if _SCALE else ["10m", "50m", "110m"]):
        if sc == "none":
            break
        try:
            ax.add_feature(cfeature.OCEAN.with_scale(sc),
                           facecolor="#0B1620", zorder=0)
            ax.add_feature(cfeature.LAND.with_scale(sc),
                           facecolor="#14202A", zorder=0)
            ax.add_feature(cfeature.LAKES.with_scale(sc),
                           facecolor="#0B1620", edgecolor=STATE,
                           linewidth=0.35, zorder=1)
            if counties and sc == "10m":
                try:
                    ax.add_feature(_counties(sc), linewidth=0.28, zorder=4)
                except Exception:
                    pass
            ax.add_feature(cfeature.STATES.with_scale(sc),
                           edgecolor=STATE, linewidth=0.6 * lw, zorder=5)
            ax.coastlines(resolution=sc, color=COAST,
                          linewidth=1.05 * lw, zorder=6)
            if _SCALE is None:
                _SCALE = sc
                if sc != "10m":
                    print(f"  note: Natural Earth 10m unavailable, using {sc}")
            break
        except Exception:
            continue
    else:
        if _SCALE is None:
            _SCALE = "none"
            print("  warning: Natural Earth shapefiles unavailable; frames "
                  "will have no map furniture.")

    if gridlines:
        try:
            gl = ax.gridlines(draw_labels=False, linewidth=0.35,
                              color=FAINT, alpha=0.55, linestyle=(0, (3, 4)),
                              zorder=7)
            gl.xlocator = plt.MultipleLocator(0.5)
            gl.ylocator = plt.MultipleLocator(0.5)
        except Exception:
            pass

    if rings and center:
        clat, clon = center
        th = np.linspace(0, 2 * np.pi, 361)
        for r_nmi in rings:
            r = r_nmi * NMI_KM
            dlat = (r / 111.19) * np.cos(th)
            dlon = (r / (111.19 * np.cos(np.radians(clat)))) * np.sin(th)
            ax.plot(clon + dlon, clat + dlat, transform=ccrs.PlateCarree(),
                    color=CYAN, linewidth=0.7, alpha=0.75,
                    linestyle=(0, (5, 3)), zorder=8)
            ax.text(clon, clat + (r / 111.19), f"{int(r_nmi)} nmi",
                    transform=ccrs.PlateCarree(), color=CYAN, fontsize=6.0,
                    ha="center", va="bottom", zorder=9,
                    family="monospace", alpha=0.9)


# ---------------------------------------------------------------------------
# annotation
# ---------------------------------------------------------------------------
def zulu(t):
    """Format a numpy datetime64 or datetime as 'YYYY-MM-DD HHMMZ'."""
    s = str(np.datetime64(t, "m")) if not isinstance(t, str) else t
    s = s.replace("T", " ")
    if s.endswith("Z"):
        s = s[:-1]
    return s[:16].replace(":", "", 1).rstrip() + "Z" if len(s) >= 16 else s + "Z"


def _tag(ax, x, y, text, ha, va, size, color, weight="normal"):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=size, color=color, fontweight=weight, zorder=20,
            family="DejaVu Sans Mono",
            bbox=dict(boxstyle="square,pad=0.32", facecolor="#0B1218",
                      edgecolor="#2B3B48", linewidth=0.6, alpha=0.82))


def annotate(ax, label, valid, init=None, lead=None, extra=None,
             members=None):
    """Burn identity and timing into the frame.

    Top-left  field name
    Top-right VALID <zulu>            <- the thing you check first
    Bot-left  INIT <zulu>  F+NNNm
    Bot-right ensemble / domain detail
    """
    _tag(ax, 0.012, 0.988, label, "left", "top", 8.4, INK, "bold")
    _tag(ax, 0.988, 0.988, f"VALID  {valid}", "right", "top", 9.0, SIGNAL,
         "bold")

    if init is not None:
        left = f"INIT {init}"
        if lead is not None:
            left += f"   F+{int(lead):03d}m"
        _tag(ax, 0.012, 0.012, left, "left", "bottom", 7.6, DIM)

    right = []
    if members:
        right.append(f"{members} mem")
    if extra:
        right.append(extra)
    if right:
        _tag(ax, 0.988, 0.012, "  ".join(right), "right", "bottom", 7.2, DIM)


def style_colorbar(cb, label, levels=None, fmt=None):
    cb.set_label(label, color=INK, fontsize=9.5, labelpad=7,
                 fontweight="bold")
    cb.ax.tick_params(colors=INK, labelsize=8.0, length=3, width=0.8)
    cb.outline.set_edgecolor("#2B3B48")
    cb.outline.set_linewidth(0.7)
    if levels is not None:
        cb.set_ticks(levels)
        if fmt:
            cb.set_ticklabels([fmt(v) for v in levels])
    for s in cb.ax.spines.values():
        s.set_edgecolor("#2B3B48")
