#!/bin/bash
set -e
cd "$(dirname "$0")/.."

ANOM_LOG="outputs/plan_b_aggregate/anomalies.log"
mkdir -p outputs/plan_b_aggregate
> "$ANOM_LOG"

log_anomaly() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$ANOM_LOG" >&2
}

echo "=== Plan B: Scene 1 — brass_goblet ==="
blender --background --python scripts/render_blender_scene_v2.py -- \
    --scene-name brass_goblet \
    --output-dir data/scene_brass_goblet_v2 \
    --samples 64 \
    || log_anomaly "brass_goblet render/experiment failed; continuing to scene 2"

echo ""
echo "=== Plan B: Scene 2 — glass_suzanne ==="
blender --background --python scripts/render_blender_scene_v2.py -- \
    --scene-name glass_suzanne \
    --output-dir data/scene_glass_suzanne \
    --samples 64 \
    || log_anomaly "glass_suzanne render/experiment failed"

echo ""
echo "=== Aggregate results ==="
if [ -d "outputs/dual_path" ]; then
    uv run python3 scripts/aggregate_scenes.py \
        --results-dir outputs/dual_path \
        --output-dir outputs/plan_b_aggregate \
        || log_anomaly "aggregate_scenes failed"
else
    log_anomaly "No dual-path output dir found; skipping aggregate"
fi

echo ""
echo "=== Writing morning report ==="
uv run python3 scripts/write_morning_report.py \
    --output-dir outputs/plan_b_aggregate \
    --scene-dirs data/scene_brass_goblet_v2 data/scene_glass_suzanne \
    --anom-log "$ANOM_LOG" \
    || log_anomaly "morning report generation failed"

echo ""
echo "=== Plan B complete ==="
echo "Check: outputs/plan_b_aggregate/morning_report.md"
[ -s "$ANOM_LOG" ] && echo "Anomalies: $ANOM_LOG"
