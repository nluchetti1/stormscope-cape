# StormScope · Cape Canaveral

NVIDIA StormScope (3 km / 10 min) ensemble nowcasts over a Cape Canaveral
window, with a static web viewer published to GitHub Pages.

**Research nowcast guidance. Not an LLCC evaluation — encodes no launch
commit criteria, and knows nothing about anvil rules, debris clouds, or
field mills.**

## Files

| File | Runs on | What it does |
|---|---|---|
| `setup_stormscope.sh` | GPU pod | Builds the venv, pins torch + natten, downloads checkpoints |
| `stormscope_cape.py` | GPU pod | `run` / `serve` / `fire` — the forecast engine |
| `verify_stormscope_cape.py` | GPU pod | Scores against observed MRMS + GLM, makes panel plots |
| `export_viewer.py` | anywhere (CPU) | NetCDF → PNG frames + `manifest.json` |
| `deploy_pages.sh` | anywhere | Export newest run, commit, push |
| `site/index.html` | browser | The viewer |

## Publishing the viewer

One-time:

1. Create the repo on GitHub and push this directory.
2. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
   Not "Deploy from a branch" — the workflow uses the Actions path.
3. Push once. `.github/workflows/deploy-pages.yml` publishes `site/`.

Your page lands at `https://<user>.github.io/<repo>/`.

Every run after that:

```bash
./deploy_pages.sh                    # newest .nc in outputs/
./deploy_pages.sh outputs/foo.nc     # a specific run
```

Tunables via environment: `ZOOM_KM`, `NBHD_KM`, `KEEP`, `SITE_DIR`, `OUT_DIR`.

## Repo size

Each run is roughly 1.5 MB for the ensemble fields at 6 lead times, and
grows with lead count and per-member frames. `--keep 10` prunes old runs,
but **git keeps deleted files in history forever** — pruning shrinks the
checkout, not the repo. If it gets heavy, squash history:

```bash
git checkout --orphan fresh && git add -A && git commit -m "reset"
git branch -D main && git branch -m main && git push -f origin main
```

GitHub Pages allows 100 GB/month of bandwidth and 10 builds/hour, and the
site is public unless you are on a paid plan. Everything here is public
NOAA data and an openly licensed model, so that is fine — but it is worth
being a deliberate choice.

## Viewer controls

| Key | Action |
|---|---|
| ← / → | Step lead time |
| ↑ / ↓ | Change field |
| Space | Play / stop the loop |
| Home / End | First / last lead time |

## Known gaps

- The GLM threshold is not yet calibrated. `glm_density` from the diffusion
  head has a small positive floor everywhere, so thresholding near zero
  marks the whole domain as lightning. Run `stormscope_cape.py run
  --glm-stats` to see the distribution and pick a defensible value.
- Observed GLM binning is 5-minute while the model steps at 10. The
  `--glm-bins 2` default sums the two bins ending at the valid time, but
  this has not been confirmed against NVIDIA's training convention. Treat
  the spatial pattern as meaningful and absolute counts as approximate.
- Point-by-point contingency scores are harsh on a generative model. The
  neighborhood probabilities are the fairer read.

## Deferred verification

`export_viewer.py` publishes a forecast immediately. `verify_pending.py`
comes back later, once the valid times have elapsed, and scores it.

It is a queue, not a fixed delay: any published run that has not been
scored and whose last valid time is more than `--min-age-min` in the past
gets picked up. A 2 h forecast is verifiable 2 h after init — you do not
have to wait a day. Running daily at 14Z is just a convenient cadence.

```bash
python verify_pending.py --dry-run     # what is pending
python verify_pending.py               # score it all
./verify_cron.sh                       # score, commit, push
```

Cron on the pod, daily at 14Z:

```
0 14 * * *  /workspace/verify_cron.sh >> /workspace/verify.log 2>&1
```

This only works while a pod is up. It needs no GPU, so a cheap CPU-only
pod is enough if you want it unattended — but the simplest thing is to run
it at the start of your next session and let it catch up.

### Metrics

| Metric | Reading |
|---|---|
| `fss_refc`, `fss_glm` | Fractions skill score at the neighborhood radius. 1 perfect, 0 no skill. **The fair read for a generative nowcast.** |
| `csi`, `pod`, `far`, `bias` | Point contingency on the ensemble mean. Included for continuity with traditional verification; harsh on displaced cells. |
| `brier_refc`, `brier_glm` | Brier score of the neighborhood probability. Lower is better. |

Why FSS matters here: a storm cell displaced 6 km scores **0.88 by FSS but
only 0.51 by point CSI**, even though the forecast is meteorologically
right. Diffusion models produce plausible structure, not the
minimum-error field, so point scores systematically understate them.

`null` in the JSON means undefined — no coverage on either side — rather
than zero skill.

### In the viewer

A verified run shows a cyan **verified** badge, per-lead FSS/CSI/Brier in
the bottom strip, and enables the **Obs** button (or `B`) to blink between
forecast and observed at the same valid time.

## Plot styling

`plotstyle.py` is the single source of truth for palettes, map furniture
and annotation, shared by the web frames and the verification panels.

- **Reflectivity** uses the standard NWS Level-3 dBZ palette on 5 dBZ
  steps (5–75). Not `turbo`: with the NWS palette you know 40 dBZ is
  orange without consulting the scale, and discrete steps make gradients
  legible as gradients.
- **Probability** uses a discrete cool-to-warm ramp at 5/10/20…100 %.
- **Flash density** uses a custom violet→ember→white-hot ramp on a log
  scale, since no community standard exists.

Every frame carries its own stamps: field name (top left), **VALID
`YYYY-MM-DD HHMMZ`** in amber (top right), init time and lead (bottom
left), ensemble size and domain (bottom right). The viewer shows this too,
but burning it in means a screenshot pulled into a briefing is unambiguous
about what it is and when it was valid.

Map furniture is coastline, state and county lines, shaded land/ocean and
a half-degree graticule, degrading 10m → 50m → 110m → none if Natural
Earth shapefiles cannot be fetched.

Range rings are off by default. Add them with:

```bash
python export_viewer.py <file.nc> --rings 5,10
```

which draws dashed 5 and 10 nmi circles around the pad — the standoff
distances the neighborhood probabilities are built around.
