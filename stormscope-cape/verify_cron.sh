#!/usr/bin/env bash
#
# Deferred verification pass. Scores any published run whose valid times
# have elapsed, then publishes the result.
#
# Nothing here starts a forecast -- it only scores forecasts that already
# exist. Safe to run on any CPU box with the venv and the .nc files.
#
# Daily at 14Z via cron:
#   0 14 * * *  /workspace/verify_cron.sh >> /workspace/verify.log 2>&1
#
# Or just run it by hand whenever you next have a box up; it catches up on
# everything pending, not only yesterday.
set -euo pipefail
cd "${WS:-/workspace}"
source ./env.sh

echo "=== verification pass $(date -u +%Y-%m-%dT%H:%MZ) ==="

python verify_pending.py \
    --site-dir site --nc-dir outputs \
    --min-age-min "${MIN_AGE_MIN:-30}" \
    --nbhd-km "${NBHD_KM:-10}" --zoom-km "${ZOOM_KM:-120}"

if [ -d .git ] && ! git diff --quiet site 2>/dev/null; then
    git add site
    git commit -m "verification: $(date -u +%Y-%m-%dT%H:%MZ)"
    git push origin main
    echo "published"
else
    echo "nothing new to publish"
fi
