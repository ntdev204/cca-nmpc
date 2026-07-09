# Human Perception Package Replan

## Overview

Build `cca_nmpc_perception`, the ROS2 human perception package that turns runtime RGB-D camera input into `cca_nmpc_msgs/HumanStateArray` for the CCA-NMPC pipeline.

The message layer is already implemented in `src/cca_nmpc_msgs` and should be treated as a completed dependency. This replan focuses only on the remaining perception package: RGB-D image input → YOLO26m TensorRT `.engine` inference → depth projection → TF transform into `map` → per-track Kalman filtering → `/human_states`.

## Current Baseline

Completed:

- `src/cca_nmpc_msgs/package.xml`
- `src/cca_nmpc_msgs/CMakeLists.txt`
- `HumanState.msg`
- `HumanStateArray.msg`
- `HumanPrediction*.msg`
- `HumanUncertainty*.msg`
- `ContextIndex*.msg`
- `AdaptiveParams.msg`
- `NmpcDiagnostics.msg`
- `docs/07_yaml_parameters.md` updated to TensorRT `.engine`, Astra topic defaults, optical-frame projection, depth-alignment requirement, and sensor QoS contract.

Assumption for this plan: `cca_nmpc_msgs` builds successfully or will be verified before integrating perception.

## Project Type

**BACKEND / ROBOTICS ROS2 package.** No web/mobile UI work. Primary implementation agent: `backend-specialist`; supporting agents: `test-engineer`, `performance-optimizer`, and `security-auditor` for model loading and runtime safety checks.

## Success Criteria

- `cca_nmpc_perception` builds as an `ament_python` ROS2 package and depends on the existing `cca_nmpc_msgs` package.
- The perception node subscribes to RGB image, depth image, camera info, and TF at runtime.
- The node loads a configured YOLO26m TensorRT `.engine`, detects people, samples aligned depth, projects detections into 3D camera coordinates, transforms to `map`, tracks people with stable `track_id`, and publishes `/human_states`.
- All tunables are declared ROS2 parameters and loaded from YAML.
- TensorRT-specific logic is isolated behind a detector interface so unit tests can use a mock detector.
- Pure logic tests cover depth projection, Kalman predict/update, track association, confidence filtering, and stale-track removal.
- `colcon build` and `colcon test` pass for `cca_nmpc_msgs` and `cca_nmpc_perception`.

## Tech Stack

- ROS2 package: `ament_python`, `rclpy`.
- Messages: existing `cca_nmpc_msgs/HumanStateArray` and `cca_nmpc_msgs/HumanState`.
- Inputs: `sensor_msgs/Image`, `sensor_msgs/CameraInfo`, TF.
- Image conversion: `cv_bridge`, OpenCV, NumPy.
- Sync: `message_filters.ApproximateTimeSynchronizer` for RGB/depth/camera info.
- QoS: sensor topics use `SensorDataQoS` (best-effort) to match the Astra publisher; publish `/human_states` reliable.
- Transform: `tf2_ros.Buffer`, `tf2_ros.TransformListener`.
- Detector: YOLO26m TensorRT `.engine` through a narrow adapter interface.
- Tracking: nearest-neighbor association plus constant-velocity Kalman filter.

## Camera Contract (Astra) — must respect

- **Optical frame, not `camera_link`.** Pinhole projection must use the frame reported by `CameraInfo.header.frame_id` / image header (an optical frame, e.g. `camera_color_optical_frame`), then TF that point into `map`. Do not project using `camera_link` — it is not axis-aligned with the optical frame.
- **Depth alignment is required.** The Astra driver defaults to `depth_registration:=false` (`src/depend/.../astra_camera/launch/astra.launch.xml`), so raw depth is not aligned to color. The launch (HP-08) must either start the camera with `depth_registration:=true` or subscribe to an explicitly aligned depth-to-color topic; the node (HP-02) must fail/warn if alignment is not guaranteed.
- **Default topics (Astra, namespaced under `/camera`):** RGB `/camera/color/image_raw`, depth `/camera/depth/image_raw` (or the aligned depth-to-color topic), camera info `/camera/color/camera_info`. These are defaults; keep them as parameters. Do not use the legacy `/camera/rgb/image_raw` names from older docs.

## File Structure

```text
src/
├── cca_nmpc_msgs/                    # already implemented; do not recreate
└── cca_nmpc_perception/
    ├── package.xml
    ├── setup.py
    ├── setup.cfg
    ├── resource/cca_nmpc_perception
    ├── cca_nmpc_perception/
    │   ├── __init__.py
    │   ├── perception_node.py
    │   ├── detector.py
    │   ├── depth_projection.py
    │   ├── kalman_filter.py
    │   └── track_manager.py
    ├── config/
    │   └── human_perception.yaml
    ├── launch/
    │   └── human_perception.launch.py
    └── test/
        ├── test_depth_projection.py
        ├── test_kalman_filter.py
        └── test_track_manager.py
```

Optional later integration, not required for first perception package milestone:

```text
src/cca_nmpc_bringup/
└── launch/
    └── perception_runtime.launch.py
```

## Task Breakdown

### HP-00 — Verify existing message package

Agent: `test-engineer`; skills: `testing-patterns`; priority: P0; dependencies: none.

INPUT → existing `src/cca_nmpc_msgs` implementation.
OUTPUT → confirmed message package baseline before perception work depends on it.
VERIFY → `colcon build --packages-select cca_nmpc_msgs` succeeds; generated Python imports for `HumanState` and `HumanStateArray` resolve.

### HP-01 — Create `cca_nmpc_perception` skeleton

Agent: `backend-specialist`; skills: `python-patterns`, `clean-code`; priority: P0; dependencies: HP-00.

INPUT → proposed package layout from `docs/02_system_architecture.md` and existing `cca_nmpc_msgs` package.
OUTPUT → `ament_python` package with console entry point `perception_node`, config folder, launch folder, and test folder.
VERIFY → `colcon build --packages-select cca_nmpc_perception` succeeds and `ros2 run cca_nmpc_perception perception_node --ros-args --help` resolves after sourcing the workspace.

### HP-02 — Add perception YAML parameters (mirror docs/07)

Agent: `backend-specialist`; skills: `clean-code`; priority: P0; dependencies: HP-01.

INPUT → the updated `perception_node` block in `docs/07_yaml_parameters.md`.
OUTPUT → `config/human_perception.yaml` with `yolo_engine_path`, Astra topic defaults (`/camera/color/image_raw`, aligned depth topic, `/camera/color/camera_info`), `sensor_qos`, `camera_optical_frame`, `map_frame`, `require_depth_alignment`, detection threshold, depth sampling radius, `max_track_age_sec`, association distance gate, and Kalman noise.
VERIFY → node declares and reads every key; startup fails clearly if `yolo_engine_path` is missing/unreadable, and warns/aborts if `require_depth_alignment` is true but the configured depth topic is not an aligned/registered stream.

### HP-03 — Detector interface + mock + TensorRT adapter

Agent: `backend-specialist`; skills: `python-patterns`, `clean-code`; priority: P1; dependencies: HP-02.

INPUT → RGB image frames and YOLO26m TensorRT `.engine` path.
OUTPUT → `detector.py` with `Detection`, `HumanDetector` interface, `MockDetector` (test-only, no TensorRT import), and `TensorRtYoloDetector` that loads the engine, runs inference, filters person class, applies confidence threshold, returns normalized boxes.
VERIFY → mock path returns deterministic boxes without importing TensorRT; TensorRT path initializes with a valid engine and raises a clear config error on an invalid/missing engine before the node spins.

### HP-04 — Depth projection (optical frame)

Agent: `backend-specialist`; skills: `python-patterns`; priority: P1; dependencies: HP-03.

INPUT → bounding boxes, aligned depth image, `CameraInfo` intrinsics.
OUTPUT → `depth_projection.py` produces a 3D point **in the optical frame** (`CameraInfo.header.frame_id`) using robust median valid depth around the box center.
VERIFY → tests cover valid depth, missing/zero depth, out-of-image boxes, and pinhole math with known intrinsics; output point is expressed in the optical frame, not `camera_link`.

### HP-05 — TF transform to map

Agent: `backend-specialist`; skills: `api-patterns`; priority: P1; dependencies: HP-04.

INPUT → optical-frame 3D point and configured `map_frame`; source frame taken from the incoming header, not a hard-coded `camera_link`.
OUTPUT → transform to `map` via TF and extract 2D `(x, y)` for the CCA-NMPC state.
VERIFY → mocked TF test confirms transform is applied from the optical frame; runtime skips detections (no publish of wrong-frame data) when TF is unavailable or times out.

### HP-06 — Kalman filter + track manager

Agent: `backend-specialist`; skills: `clean-code`; priority: P1; dependencies: HP-05.

INPUT → timestamped map-frame detections with confidence.
OUTPUT → `kalman_filter.py` constant-velocity filter (`x, y, vx, vy`) and `track_manager.py` nearest-neighbor association with persistent `track_id`, new-track creation, stale pruning after `max_track_age_sec`, and confidence propagation.
VERIFY → tests cover predict/update with variable `dt`, velocity convergence, track ID persistence, nearby/crossing association within the gate, new-track entry, stale removal, and empty-detection frames.

### HP-07 — Implement `perception_node.py`

Agent: `backend-specialist`; skills: `api-patterns`, `clean-code`; priority: P1; dependencies: HP-02 through HP-06.

INPUT → synchronized RGB/depth/camera-info (SensorDataQoS), TF, detector, projector, tracker.
OUTPUT → ROS2 node publishing `cca_nmpc_msgs/HumanStateArray` on `/human_states`; each `HumanState` carries `header, track_id, x, y, vx, vy, confidence` (per-message `header.stamp` = detector observation time); `HumanStateArray.header` carries output frame (`map`) and the most-recent observation time. Note: `HumanState` uses `std_msgs/Header header`, NOT a bare `stamp` field.
VERIFY → with `MockDetector` and synthetic image/depth/camera-info, a node-level or launch smoke test publishes a valid `/human_states` without TensorRT hardware.

### HP-08 — Launch integration (camera + alignment)

Agent: `devops-engineer`; skills: `deployment-procedures`, `bash-linux`; priority: P2; dependencies: HP-07.

INPUT → `human_perception.yaml`, Astra launch (`astra.launch.xml`, defaults `depth_registration:=false`), perception entry point.
OUTPUT → `launch/human_perception.launch.py` starts perception with param overrides; when it also brings up the camera it must set `depth_registration:=true` (or wire an explicit aligned depth-to-color topic) so projection input is aligned.
VERIFY → `ros2 launch cca_nmpc_perception human_perception.launch.py` starts the node; `ros2 node info` shows the expected subscriptions/publisher; depth input is confirmed aligned.

### HP-09 — Runtime validation, performance, and safety

Agent: `performance-optimizer` + `security-auditor`; skills: `performance-profiling`, `vulnerability-scanner`; priority: P2; dependencies: HP-08.

INPUT → running Astra camera, valid YOLO26m `.engine`, perception node.
OUTPUT → per-stage timing logs (callback, inference, projection, TF, tracking, publish); guarded model-file handling with no hard-coded machine-local paths.
VERIFY → average perception cycle supports target camera FPS and identifies the bottleneck stage; missing/unreadable engine, invalid frames, unaligned depth, and invalid thresholds produce clear errors or warnings rather than silent bad output.

## Phase X: Final Verification

- Run `colcon build --symlink-install --packages-select cca_nmpc_msgs cca_nmpc_perception`.
- Run `colcon test --packages-select cca_nmpc_msgs cca_nmpc_perception`.
- Run `colcon test-result --verbose` and confirm zero failures.
- Run `python .agents/skills/lint-and-validate/scripts/lint_runner.py .` and review output before fixing anything.
- Mock runtime smoke test: launch perception with `MockDetector`, publish synthetic RGB/depth/camera-info using SensorDataQoS, and confirm `/human_states` publishes a valid `HumanStateArray`.
- Frame smoke test: confirm projected points originate in `CameraInfo.header.frame_id` / optical frame and are TF-transformed into `map` before publish.
- Hardware runtime smoke test: start Astra camera with depth registration enabled or aligned-depth topic configured, launch perception with the TensorRT engine, confirm stable `track_id` for a visible person.
- Rosbag smoke test: record `/human_states`, camera topics, `/tf`, and `/tf_static`; confirm timestamps, QoS compatibility, frame IDs, and depth alignment are consistent.

## Notes and Risks

- Do not recreate or rename `cca_nmpc_msgs`; perception should depend on the implementation already present.
- TensorRT/CUDA availability is platform-specific. Keep TensorRT imports isolated so tests and non-GPU development can run with `MockDetector`.
- The `.engine` is GPU/TensorRT-version specific and must be built on (or for) the target device; do not commit a binary engine as if it were portable.
- Project into the optical frame (`CameraInfo.header.frame_id`), never `camera_link` — the two are not axis-aligned, and using `camera_link` silently produces wrong positions.
- Astra defaults to `depth_registration:=false`, so depth is not aligned to color out of the box. Alignment is a hard prerequisite: enable hardware D2C or subscribe to an aligned depth-to-color topic before trusting projected positions.
- Sensor topic QoS must match the Astra publisher (SensorDataQoS / best-effort); a plain reliable subscription can silently receive nothing or add latency.
- This package stops at `/human_states`. It must not implement LSTM prediction, context estimation, adaptive params, or NMPC control.
