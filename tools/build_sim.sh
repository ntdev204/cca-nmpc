#!/usr/bin/env bash
# Build simulation-specific packages (robot, URDF, dependencies)

set -e

cd "$(dirname "$0")/.."

colcon build --packages-select \
  serial \
  rai_robot_urdf \
  turn_on_rai_robot \
  rai_robot_slam \
  rai_robot_keyboard \
  --symlink-install

echo "✓ Simulation packages built"
