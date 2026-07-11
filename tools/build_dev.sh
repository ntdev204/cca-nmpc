#!/usr/bin/env bash
# Targeted colcon build for CCA-NMPC core packages (fast dev iteration)

set -e

cd "$(dirname "$0")/.."

colcon build --packages-select \
  cca_nmpc_msgs \
  cca_nmpc_perception \
  cca_nmpc_context \
  cca_nmpc_adaptive_params \
  cca_nmpc_prediction \
  cca_nmpc_control \
  cca_nmpc_bringup \
  --symlink-install

echo "✓ CCA-NMPC core packages built"
