# CCA-NMPC — ROS2 Topics

All topics use `cca_nmpc_msgs` custom message types unless a standard ROS2 type is noted. QoS profiles follow ROS2 defaults (`Reliable`, `KeepLast(10)`) unless otherwise specified — sensor-rate topics use `SensorDataQoS` (best-effort) to avoid blocking on network hiccups.

---

## 1. Perception

| Topic | Message Type | Publisher | Subscriber(s) | QoS | Rate |
|---|---|---|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | Astra camera driver | `perception_node` | SensorDataQoS | camera FPS |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Astra camera driver | `perception_node` | SensorDataQoS | camera FPS |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | Astra camera driver | `perception_node` | SensorDataQoS | camera FPS |
| `/human_states` | `cca_nmpc_msgs/HumanStateArray` | `perception_node` | `prediction_node`, `context_node` | Reliable, KeepLast(10) | $f_{perception}$ |

> **Camera topics follow the Astra driver** (`src/depend/astra_camera_ros2`, `astra.launch.xml`), namespaced under `/camera`, matching `perception_node` defaults in `07_yaml_parameters.md`. Do **not** use the legacy `/camera/rgb/image_raw` name. Depth must be aligned to color (launch with `depth_registration:=true` or subscribe to an aligned depth-to-color topic); pinhole projection uses the optical frame from `CameraInfo.header.frame_id`, not `camera_link`.

## 2. Prediction

| Topic | Message Type | Publisher | Subscriber(s) | QoS | Rate |
|---|---|---|---|---|---|
| `/human_predictions` | `cca_nmpc_msgs/HumanPredictionArray` | `prediction_node` | `context_node`, `nmpc_controller_node` | Reliable, KeepLast(5) | $f_{LSTM}$ |
| `/human_pred_uncertainty` | `cca_nmpc_msgs/HumanUncertaintyArray` | `prediction_node` | `context_node` | Reliable, KeepLast(5) | $f_{LSTM}$ |

## 3. Context & Adaptive Parameters

| Topic | Message Type | Publisher | Subscriber(s) | QoS | Rate |
|---|---|---|---|---|---|
| `/context_index` | `cca_nmpc_msgs/ContextIndexArray` | `context_node` | `adaptive_param_node`, `nmpc_controller_node` (diagnostics) | Reliable, KeepLast(10) | $f_{context}$ (= NMPC rate) |
| `/adaptive_params` | `cca_nmpc_msgs/AdaptiveParams` | `adaptive_param_node` | `nmpc_controller_node` | Reliable, KeepLast(10) | $f_{context}$ |

## 4. Control

| Topic | Message Type | Publisher | Subscriber(s) | QoS | Rate |
|---|---|---|---|---|---|
| `/odom_combined` | `nav_msgs/Odometry` | robot EKF / base driver | `context_node`, `nmpc_controller_node` | SensorDataQoS | odom rate |
| `/goal_pose` | `geometry_msgs/PoseStamped` | mission manager | `nmpc_controller_node` | Reliable | event |
| `/reference_path` | `nav_msgs/Path` | Semantic A*/Nav2 planner | `nmpc_controller_node` | Reliable | planner rate |
| `/local_costmap/costmap` | `nav2_msgs/Costmap` | Nav2 costmap server | `nmpc_controller_node` | Reliable | costmap update rate |
| `/cmd_vel` | `geometry_msgs/Twist` | `nmpc_controller_node` | robot base driver | Reliable, KeepLast(1) | $f_{NMPC}$ |
| `/nmpc_diagnostics` | `cca_nmpc_msgs/NmpcDiagnostics` | `nmpc_controller_node` | logging / rosbag, RViz overlay | Reliable, KeepLast(50) | $f_{NMPC}$ |

## 5. TF

| Frame pair | Broadcaster | Notes |
|---|---|---|
| `map` → `odom_combined` | localization stack (e.g., AMCL / SLAM / slam_toolbox) | global localization relation used by the current robot stack |
| `odom_combined` → `base_footprint` | robot base driver / EKF | wheel/IMU odometry fused into the robot base frame |
| `base_footprint` → `base_link` | static TF / robot description | body-frame bridge used by the current robot stack |
| `base_link` → `camera_link` | static TF (URDF) | Mini Mecanum mount: camera centre at 0.21 m above ground, pitched upward 20° (`pitch = -0.34906585 rad` in ROS coordinates) |

---

## 6. Topic Naming Convention
- All custom topics are namespaced under the robot's namespace when multi-robot deployment is used, e.g. `/robot1/human_states`.
- Diagnostic/debug topics are prefixed under `/cca_nmpc/debug/...` when added (e.g., `/cca_nmpc/debug/phi_history` for a rolling buffer used in RViz plotting), to keep them clearly separated from the core control-loop topics above.
