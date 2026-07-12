---
name: env-toolchain
description: Windows vs WSL toolchain split for building/testing this ROS2 CCA-NMPC repo
metadata:
  type: project
---

Windows (dev + offline unit tests) has the FULL offline Python stack: Python 3.14, numpy 2.4, pytest 9, casadi 3.7.2, torch 2.11+cpu, onnx 1.21. Run all ROS-free core `pytest` here.

WSL Ubuntu (ROS2 Humble at /opt/ros/humble, Python 3.10, numpy 1.21) is the ONLY place for `colcon build --symlink-install` and `colcon test`. WSL has NO casadi and NO torch — so any test that imports casadi/torch must run on Windows pytest, NOT under colcon. Keep ROS-free cores importable without casadi/torch so colcon lint/test of ROS packages doesn't pull heavy deps.

**Why:** Plans specify two-tier verification: "VERIFY (now)" = Windows pytest, "VERIFY (later)" = WSL colcon. Matches available tooling.

**How to apply:** Develop code on Windows (path D:\Research\cca-nmpc). Unit-test cores on Windows. For colcon: `wsl bash -c "cd ~/cca-nmpc && git pull ... && source /opt/ros/humble/setup.bash && colcon build/test"`. WSL repo is at ~/cca-nmpc, synced via git (Windows remote name is `cca-nmpc`, push to develop; WSL remote is `origin`). See [[wsl-sync-workflow]].
