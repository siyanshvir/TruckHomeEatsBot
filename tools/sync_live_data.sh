#!/bin/sh
# Copy the live bot's data into this folder before the finals bot takes over. Read-only on the live folder.
# Refuses to run while this folder's bot is running (a recent connection-status.json), so it can never
# overwrite newer finals-side learning.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LIVE="${LIVE_DATA:-$HERE/../TruckHomeEatsBot/data}"
DST="$HERE/data"
[ -d "$LIVE" ] || { echo "live data folder not found: $LIVE"; exit 1; }
mkdir -p "$DST/live-observations" "$DST/reviews"
STATUS="$DST/connection-status.json"
if [ -f "$STATUS" ] && [ -z "$(find "$STATUS" -mmin +2 2>/dev/null)" ]; then
  echo "the finals bot looks like it is running (fresh connection-status.json); not syncing"; exit 1
fi
for f in runtime-v1.json evaluation-v2.jsonl forecasts-v2.jsonl decisions-v1.jsonl collect.jsonl; do
  if [ -f "$LIVE/$f" ] && { [ ! -f "$DST/$f" ] || [ "$LIVE/$f" -nt "$DST/$f" ]; }; then cp "$LIVE/$f" "$DST/$f"; echo "synced $f"; fi
done
cp -n "$LIVE"/live-observations/*.json "$DST/live-observations/" 2>/dev/null || true
echo "live-observations: $(ls "$DST/live-observations" | wc -l | tr -d ' ') files"
python3 - "$DST/runtime-v1.json" <<'PY'
import json, sys
brain = json.load(open(sys.argv[1]))['brain']
print('checkpoint learned through day', brain.get('seen'), '| market', brain.get('market'))
PY
