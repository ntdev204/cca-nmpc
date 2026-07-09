# CCA-NMPC — ROS2 Topics

All topics use `cca_nmpc_msgs` custom message types unless a standard ROS2 type is noted. QoS profiles follow ROS2 defaults (`Reliable`, `KeepLast(10)`) unless otherwise specified — sensor-rate topics use `SensorDataQoS` (best-effort) to avoid blocking on network hiccups.

---

## 1. Perception

| Topic | Message Type | Publisher | Subscriber(s) | QoS | Rate |
|---|---|---|---|---|---|
| `/camera/rgb/image_raw` | `sensor_msgs/Image` | camera driver | `perception_node` | SensorDataQoS | camera FPS |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | camera driver | `perception_node` | SensorDataQoS | camera FPS |
| `/human_states` | `cca_nmpc_msgs/HumanStateArray` | `perception_node` | `prediction_node`, `context_node` | Reliable, KeepLast(10) | $f_{perception}$ |

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
| `/odom` | `nav_msgs/Odometry` | robot base driver | `context_node`, `nmpc_controller_node` | SensorDataQoS | odom rate |
| `/local_costmap/costmap` | `nav2_msgs/Costmap` | Nav2 costmap server | `nmpc_controller_node` | Reliable | costmap update rate |
| `/cmd_vel` | `geometry_msgs/Twist` | `nmpc_controller_node` | robot base driver | Reliable, KeepLast(1) | $f_{NMPC}$ |
| `/nmpc_diagnostics` | `cca_nmpc_msgs/NmpcDiagnostics` | `nmpc_controller_node` | logging / rosbag, RViz overlay | Reliable, KeepLast(50) | $f_{NMPC}$ |

## 5. TF

| Frame pair | Broadcaster | Notes |
|---|---|---|
| `map` → `odom` | localization stack (e.g., AMCL / SLAM) | standard Nav2 convention |
| `odom` → `base_link` | robot base driver | wheel/IMU odometry |
| `base_link` → `camera_link` | static TF (URDF) | fixed camera mount |

---

## 6. Topic Naming Convention
- All custom topics are namespaced under the robot's namespace when multi-robot deployment is used, e.g. `/robot1/human_states`.
- Diagnostic/debug topics are prefixed under `/cca_nmpc/debug/...` when added (e.g., `/cca_nmpc/debug/phi_history` for a rolling buffer used in RViz plotting), to keep them clearly separated from the core control-loop topics above.
