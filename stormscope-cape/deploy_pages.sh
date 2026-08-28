#!/usr/bin/env bash
#
# Export the newest StormScope run(s) and push the viewer to GitHub Pages.
#
#   ./deploy_pages.sh                       # newest .nc in outputs/
#   ./deploy_pages.sh outputs/foo.nc        # a specific file
#
# CPU only. Safe to run locally after pulling .nc files down, or on the pod.
set -euo pipefail

SITE_DIR="${SITE_DIR:-site}"
OUT_DIR="${OUT_DIR:-outputs}"
KEEP="${KEEP:-10}"          # runs retained; older ones are pruned
MEMBERS="${MEMBERS:-all}"   # per-member frames: all, 0,1,2, or empty for none
ZOOM_KM="${ZOOM_KM:-120}"
NBHD_KM="${NBHD_KM:-10}"

if [ $# -gt 0 ]; then
    FILES=("$@")
else
    mapfile -t FILES < <(ls -t "$OUT_DIR"/stormscope_cape_*.nc 2>/dev/null | head -1)
fi
if [ ${#FILES[@]} -eq 0 ] || [ -z "${FILES[0]:-}" ]; then
    echo "No NetCDF files found in $OUT_DIR/. Run stormscope_cape.py first."
    exit 1
fi

echo "Exporting: ${FILES[*]}"
python export_viewer.py "${FILES[@]}" \
    --site-dir "$SITE_DIR" --zoom-km "$ZOOM_KM" --nbhd-km "$NBHD_KM" \
    --members "$MEMBERS" --keep "$KEEP" --clean

if [ ! -d .git ]; then
    echo
    echo "Not a git repo yet. One-time setup:"
    echo "  git init -b main"
    echo "  git remote add origin git@github.com:<you>/stormscope-cape.git"
    exit 0
fi

git add "$SITE_DIR"
if git diff --cached --quiet; then
    echo "No changes to publish."
    exit 0
fi

git commit -m "viewer: $(date -u +%Y-%m-%dT%H:%MZ) run export"
git push origin main
echo
echo "Pushed. Actions will publish in ~1 min:"
echo "  https://github.com/<you>/<repo>/actions"
