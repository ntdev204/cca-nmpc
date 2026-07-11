# Review 2 — Đánh giá lại repository CCA-NMPC

Task gốc: **“Review lại 1 lần nữa”**

Mục tiêu review:

- Kiểm tra implementation hiện tại đã tới đâu.
- Đối chiếu code với mô hình toán CCA-NMPC và các tài liệu đặc tả `01`–`08`.
- Kiểm tra xung đột giữa code, docs, topic, message, YAML và solver.
- Review sâu nhất phần mô hình toán và dataset.

Kết luận ngắn:

> Repository đã tiến bộ rõ so với bản trước, đặc biệt ở phần per-human context, adaptive parameters, solver constraint và dataset preprocessing. Tuy nhiên hệ thống **chưa xong**, **chưa sẵn sàng thu dataset thật nghiêm túc**, và **chưa sẵn sàng chạy robot thật/IJAT experiment**. Blocker lớn nhất hiện tại nằm ở perception tracking: stale track không bị xóa đúng, có thể tạo pseudo-ground-truth sai cho dataset.

---

## 1. Trạng thái kiểm thử

### 1.1 Dataset / training / context calibration

Lệnh kiểm tra:

```powershell
python -m pytest -q tools\lstm_dataset tools\lstm_training tools\context_calibration
```

Kết quả:

```text
53 passed
```

Nhận xét:

- Pipeline dataset đã có test khá tốt.
- Split theo trajectory key đã đúng hướng.
- Resample và normalization đã tiến bộ.
- Chưa có logger dataset thật hoàn chỉnh.

### 1.2 ROS-side Python tests

Lệnh kiểm tra:

```powershell
python -m pytest -q src\cca_nmpc_control\cca_nmpc_control\nmpc_solver\tests src\cca_nmpc_control\test src\cca_nmpc_context\test src\cca_nmpc_adaptive_params\test src\cca_nmpc_prediction\test src\cca_nmpc_perception\test
```

Kết quả:

```text
137 passed
2 failed
```

Hai test lỗi:

```text
src\cca_nmpc_perception\test\test_track_manager.py::TestStalePruning::test_stale_track_removed_after_max_age
src\cca_nmpc_perception\test\test_track_manager.py::TestStalePruning::test_only_stale_tracks_removed
```

Nguyên nhân chính:

> Track cũ không bị prune vì bước `predict()` cập nhật `last_update_time`, làm hệ thống tưởng track vẫn còn mới dù không có measurement thật.

Đây là lỗi nghiêm trọng vì ảnh hưởng trực tiếp tới dataset thật.

### 1.3 Build ROS2 bằng WSL

Lệnh kiểm tra:

```bash
colcon build --packages-select cca_nmpc_msgs cca_nmpc_context cca_nmpc_adaptive_params cca_nmpc_prediction cca_nmpc_perception cca_nmpc_control
```

Kết quả:

```text
6 packages finished successfully
```

Nhận xét:

- Build được.
- Message/package dependency đủ để build.
- Nhưng build thành công chưa đồng nghĩa runtime đúng.

### 1.4 Ruff lint

Lệnh kiểm tra:

```powershell
python -m ruff check src tools
```

Kết quả:

```text
154 errors
```

Phần lớn là:

- `E702` nhiều statement trên một dòng.
- `F401` unused import.
- `F841` unused variable.
- `E402` import không ở đầu file.

Nhận xét:

- Đây chưa phải blocker mô hình toán.
- Nhưng trước khi freeze/release cần dọn.

### 1.5 Git diff check

Lệnh kiểm tra:

```powershell
git diff --check
```

Kết quả:

```text
Passed, chỉ có cảnh báo LF -> CRLF
```

---

## 2. Traceability matrix

| Requirement | Code/docs hiện tại | Status |
|---|---|---|
| Per-human `phi_j` | `cca_nmpc_context`, `ContextIndexArray`, adaptive params | Đạt phần chính |
| Per-human `d_safe(phi_j)` | `AdaptiveParams.msg` có `d_safe_per_human`; controller truyền sang solver | Đạt |
| Human constraint per-human | `casadi_solver.py` tạo constraint cho từng human | Đạt |
| Human cost per-human | `casadi_solver.py` tính `J_human` theo từng human | Đạt |
| Constraint kiểm tra toàn horizon | solver dùng stages `k=0..N` | Đạt |
| `Q_diag` kích thước 3 | message/YAML đã về `[3]` | Đạt |
| Adaptive velocity clamp | adaptive node tính global velocity bounds theo `phi_aggregate_used` | Đạt cơ bản |
| EMA vs dwell-time | đã tách EMA và dwell gate rõ hơn | Đạt |
| Uncertainty stale growth | prediction uncertainty có aging/growth | Đạt cơ bản |
| Fixed human slots | solver có max humans/fixed slots | Đạt cơ bản |
| Slack per-human | solver có slack và diagnostics | Đạt cơ bản |
| Costmap/obstacle cost | `obstacle_cost = 0.0`, chưa có costmap thực | Chưa xong |
| Semantic A* / `P_ref` | controller vẫn tạo line reference bằng `np.linspace` | Chưa khớp đặc tả |
| acados backend | stub `NotImplementedError` | Chưa xong |
| Dataset schema | CSV/NPZ schema đã tốt hơn | Đạt cơ bản |
| Dataset logger thật | chưa có logger hoàn chỉnh | Chưa xong |
| LSTM split leakage | split theo trajectory key | Đạt |
| Velocity derivation | có xử lý trong dataset pipeline/perception | Đạt một phần |
| Pseudo-ground-truth risk | stale track bug còn tồn tại | Blocker |

---

## 3. Findings theo severity

## P1 — Blocker

### P1.1. Stale track không bị xóa đúng, có thể làm sai dataset thật

File liên quan:

```text
src/cca_nmpc_perception/cca_nmpc_perception/kalman_filter.py
src/cca_nmpc_perception/cca_nmpc_perception/track_manager.py
```

Vấn đề:

- `TrackManager` gọi `predict()` cho các track.
- `predict()` cập nhật `last_update_time`.
- Sau đó `_prune_stale_tracks()` dùng `current_time - last_update_time`.
- Vì `last_update_time` vừa bị predict cập nhật, track cũ không bao giờ bị xem là stale.

Tác động:

- Người đã rời khỏi camera vẫn có thể tồn tại trong `HumanState`.
- Logger sẽ ghi pseudo-trajectory không có measurement thật.
- LSTM có thể học dữ liệu giả.
- Context và NMPC có thể phản ứng với người không còn tồn tại.

Đây là blocker lớn nhất cho dataset.

Cách sửa:

- Tách rõ:

```text
last_prediction_time
last_measurement_time
```

- `predict()` chỉ cập nhật `last_prediction_time`.
- `update()` khi có detection thật mới cập nhật `last_measurement_time`.
- `_prune_stale_tracks()` phải dùng `last_measurement_time`.

---

### P1.2. Controller callback `_latest_humans` sai kiểu dữ liệu

File liên quan:

```text
src/cca_nmpc_control/cca_nmpc_control/nmpc_controller_node.py
```

Vấn đề:

- `_latest_humans` được khai báo như dict.
- Nhưng callback `_on_human_states` lại gán nguyên `HumanStateArray`.

Tác động:

- Hiện có thể chưa nổ vì chưa dùng sâu.
- Nhưng khi controller cần human state trực tiếp, lỗi kiểu dữ liệu sẽ xuất hiện.
- Đây là lỗi logic interface giữa message và controller.

Cách sửa:

Hoặc bỏ `_latest_humans` nếu không dùng, hoặc chuẩn hóa:

```python
self._latest_humans = {
    h.track_id: h
    for h in msg.humans
}
```

---

### P1.3. TensorRT runtime dùng CUDA stream không an toàn

File liên quan:

```text
src/cca_nmpc_perception/cca_nmpc_perception/detector.py
src/cca_nmpc_prediction/cca_nmpc_prediction/lstm_infer.py
```

Vấn đề:

- Code tạo stream tạm bằng `cuda.Stream().handle`.
- Execute async nhưng copy output host/device không có synchronization rõ ràng.
- Stream object có thể bị garbage collected.

Tác động:

- Output YOLO/LSTM có thể không ổn định.
- Lỗi runtime có thể xuất hiện ngẫu nhiên.
- Khó debug khi chạy Jetson.

Cách sửa:

- Tạo persistent stream trong constructor:

```python
self._stream = cuda.Stream()
```

- Dùng async copy với stream.
- Gọi:

```python
self._stream.synchronize()
```

trước khi đọc output.

---

### P1.4. TensorRT engine builder thiếu optimization profile cho dynamic batch

File liên quan:

```text
tools/lstm_training/export.py
tools/lstm_training/build_engine.py
```

Vấn đề:

- ONNX export dùng dynamic batch axis.
- TensorRT builder chưa thấy profile min/opt/max shape tương ứng.

Tác động:

- Build engine có thể fail hoặc engine không chạy đúng shape.
- Online LSTM inference trên Jetson có rủi ro.

Cách sửa:

- Thêm optimization profile:

```text
min = (1, T, F)
opt = (B_opt, T, F)
max = (B_max, T, F)
```

---

## P2 — Major

### P2.1. acados backend vẫn là stub

File liên quan:

```text
src/cca_nmpc_control/cca_nmpc_control/nmpc_solver/acados_solver.py
```

Vấn đề:

- Backend acados chưa implement.
- Các method còn `NotImplementedError`.

Tác động:

- Đặc tả solver nói CasADi debug, acados deployment.
- Hiện hệ thống chỉ có CasADi/IPOPT.
- Chưa đạt mục tiêu real-time deployment.

Cách sửa:

- Implement acados OCP theo cùng interface.
- Mapping parameter:
  - reference trajectory,
  - human predictions,
  - per-human `d_safe`,
  - velocity bounds,
  - `Q_diag`,
  - slack.

---

### P2.2. Controller chưa dùng Semantic A* / `P_ref`, vẫn tạo straight-line reference

File liên quan:

```text
src/cca_nmpc_control/cca_nmpc_control/nmpc_controller_node.py
```

Vấn đề:

- Đặc tả yêu cầu:

```text
Semantic A* -> P_ref -> CA-NMPC
```

- Nhưng controller hiện tạo reference bằng:

```python
np.linspace(current_pose, goal_pose, N + 1)
```

Tác động:

- Chưa khớp architecture.
- Chưa kiểm tra được tracking theo `P_ref` thật.
- Chưa có tương tác với global planner/costmap.

Cách sửa:

- Thêm subscriber cho reference path/trajectory.
- Convert path thành:

```text
mathcal{P}_ref = {P_ref,k ... P_ref,k+N}
```

- Chỉ fallback straight-line khi debug.

---

### P2.3. Costmap/obstacle cost chưa implement

File liên quan:

```text
src/cca_nmpc_control/cca_nmpc_control/nmpc_solver/casadi_solver.py
```

Vấn đề:

- Diagnostics có `cost_obstacle` hoặc `cost_costmap`.
- Nhưng solver hiện để:

```python
obstacle_cost = 0.0
```

Tác động:

- Chưa có tránh vật cản tĩnh trong local NMPC cost.
- Semantic/global planning có thể tránh đường đi tổng quát, nhưng solver local chưa có costmap gradient.

Cách sửa:

- Nếu vẫn theo thiết kế “soft cost only”, cần thêm differentiable obstacle/costmap cost.
- Tránh đưa raw Nav2 costmap không trơn trực tiếp vào nonlinear constraint.

---

### P2.4. Chưa có dataset logger thật

File liên quan:

```text
tools/lstm_dataset/
src/cca_nmpc_perception/
```

Vấn đề:

- Dataset schema/preprocessing đã có.
- Nhưng chưa có node/script rõ ràng để ghi:

```text
HumanStateArray -> CSV dataset
```

Tác động:

- Chưa thể bắt đầu thu dataset thật một cách chuẩn hóa.
- Dễ thu sai schema hoặc thiếu metadata.

Cách sửa:

- Thêm `human_trajectory_logger.py`.
- Subscribe `HumanStateArray`.
- Ghi CSV theo schema:

```text
session_id, sequence_id, timestamp, track_id, x, y, vx, vy, confidence
```

- Ghi metadata:

```text
camera model
fps
frame_id
environment
scenario
operator
date
```

---

### P2.5. Mock fallback có thể che lỗi production

File liên quan:

```text
src/cca_nmpc_perception/cca_nmpc_perception/perception_node.py
src/cca_nmpc_prediction/cca_nmpc_prediction/prediction_node.py
```

Vấn đề:

- Khi TensorRT/engine unavailable, node có thể fallback sang mock.

Tác động:

- Khi chạy thật, hệ thống có thể tưởng đang dùng YOLO/LSTM thật nhưng thực tế đang dùng mock.
- Kết quả thí nghiệm/dataset có thể mất giá trị.

Cách sửa:

- Thêm tham số rõ:

```text
allow_mock_detector
allow_mock_predictor
```

- Production mode phải fail fast nếu model thật không load được.

---

## 4. Các điểm đã đúng hoặc đã cải thiện

### 4.1. Per-human context đã đúng hướng

Hệ thống hiện không còn chỉ dùng một `phi` tổng hợp cho mọi thứ.

Đúng hướng hiện tại:

```text
phi_j -> d_safe_j
phi_aggregate_used -> Q, velocity bounds
```

Đây là nhất quán với mô hình toán mới.

### 4.2. Human cost và constraint đã per-human

Solver đã đi đúng hướng:

```text
J_human = sum_j w_h phi_j max(0, d0 - d_j)^2
```

và:

```text
d_j + s_j >= d_safe(phi_j)
```

Điều này sửa được lỗi logic cũ: cost chỉ theo người gần nhất nhưng constraint lại per-human.

### 4.3. Constraint đã kiểm tra toàn horizon

Constraint human không còn chỉ kiểm tra terminal stage.

Đây là điểm rất quan trọng đối với né người động.

### 4.4. `Q_diag` đã đúng kích thước 3

Message hiện dùng:

```text
float64[3] q_diag
```

phù hợp với:

```text
P_r = [x_r, y_r, theta_r]^T
```

Không còn lỗi `Q_diag[9]`.

### 4.5. EMA và dwell-time đã tách rõ hơn

Pipeline context hiện đúng tinh thần:

```text
phi_raw
  -> EMA
  -> dwell gate
  -> phi_used
```

EMA xử lý nhiễu.

Dwell-time giới hạn tốc độ thay đổi tham số thích nghi.

### 4.6. Uncertainty stale growth đã có

Prediction uncertainty hiện có aging/stale growth.

Đây là phù hợp với đặc tả:

```text
prediction càng cũ -> uncertainty càng tăng
```

### 4.7. Dataset split đã chống leakage tốt hơn

Dataset split theo composite trajectory key:

```text
(session_id, sequence_id, track_id)
```

Đây là đúng. Không nên split ngẫu nhiên từng frame.

### 4.8. Resample và normalization đã tốt hơn

Pipeline dataset hiện có:

- resample cố định tần số,
- gap flag,
- window rejection,
- normalization thống nhất.

Điểm cần giữ:

> Không được để train/val/test cùng lấy các window từ cùng một trajectory gốc.

---

## 5. Review sâu phần mô hình toán

## 5.1. Mô hình context hiện tại

Mô hình mới không còn nên dùng OZ/NC/HPZ như logic điều khiển chính.

Logic đúng:

```text
per-human phi_j
```

và:

```text
d_safe_j = d_safe(phi_j)
```

Global adaptation:

```text
phi_aggregate_used = max_j(phi_j_used)
```

được dùng cho:

```text
Q(phi)
v_max(phi)
omega_max(phi)
```

Đây là hợp lý vì:

- safety distance là quan hệ robot-người, nên per-human;
- velocity/Q là hành vi tổng thể của robot, nên dùng aggregate risk.

## 5.2. Human cost/constraint

Hiện đã đúng hướng:

```text
J_human = sum_j ...
```

và constraint per-human.

Tuy nhiên cần lưu ý differentiability:

```text
max(0, d0 - d_j)^2
```

là hinge cost, không trơn tại biên.

Với CasADi/IPOPT có thể chạy, nhưng khi chuyển acados cần cân nhắc:

- softplus approximation,
- squared slack,
- hoặc smooth hinge.

## 5.3. Costmap/obstacle

Hiện chưa implement.

Nếu dùng raw costmap thì không nên đưa trực tiếp vào hard nonlinear constraint vì costmap dạng grid không trơn.

Khuyến nghị đúng đặc tả:

```text
costmap/obstacle = soft cost
```

và nếu cần gradient, dùng:

- signed distance field,
- smoothed costmap,
- bilinear interpolation,
- hoặc differentiable obstacle primitives.

## 5.4. Solver parameter flow

Luồng đúng hiện tại:

```text
ContextNode
  -> AdaptiveParamNode
  -> AdaptiveParams
  -> Solver
```

Solver không nên tự tính:

```text
d_safe(phi)
Q(phi)
v_max(phi)
```

Solver chỉ nhận runtime parameter.

Thiết kế hiện tại đã theo hướng đó.

---

## 6. Review sâu phần dataset

## 6.1. Dataset 01 Human Trajectory

Mục tiêu dataset:

```text
past trajectory -> future trajectory
```

Không phải detection dataset.

Không phải semantic context dataset.

Schema nên giữ:

```text
session_id
sequence_id
timestamp
track_id
x
y
vx
vy
confidence
```

## 6.2. NPZ là gì?

`.npz` chỉ là định dạng NumPy để lưu dataset đã xử lý.

Nó có thể chứa:

```text
X_past
Y_future
track_id
sequence_id
normalization_stats
```

`.npz` không phải context rời rạc.

Nếu thấy `.npz` trong dataset pipeline thì bình thường.

Vấn đề chỉ xảy ra nếu trong dữ liệu hoặc model vẫn dùng OZ/NC/HPZ làm nhãn điều khiển chính.

## 6.3. OZ/NC/HPZ nếu còn tồn tại thì sao?

Theo mô hình toán mới, OZ/NC/HPZ không được là input điều khiển chính nữa.

Chỉ chấp nhận nếu chúng là:

- tên scenario thí nghiệm,
- legacy note,
- mô tả môi trường trong dataset metadata.

Không chấp nhận nếu chúng xuất hiện trong:

- context logic,
- adaptive parameter logic,
- solver parameter,
- YAML điều khiển,
- cost/constraint switching.

Mô hình toán mới phải dùng:

```text
phi_j
phi_aggregate_used
d_safe(phi_j)
Q(phi_aggregate)
v_max(phi_aggregate)
omega_max(phi_aggregate)
```

## 6.4. Pseudo-ground-truth risk

Đây là rủi ro dataset lớn nhất hiện tại.

Nếu perception tracker giữ track cũ bằng prediction nhưng không có measurement thật, logger có thể ghi dữ liệu giả.

Điều này làm dataset bị nhiễm pseudo-ground-truth.

Trước khi thu dataset thật, bắt buộc sửa stale pruning.

## 6.5. Velocity derivation

Velocity có thể tính từ:

```text
vx = (x_k - x_{k-1}) / dt
vy = (y_k - y_{k-1}) / dt
```

Nhưng cần:

- timestamp thật,
- resample ổn định,
- smoothing nhẹ,
- loại window có gap/missing track.

Không nên tính velocity trên track đã bị predict-only quá lâu.

## 6.6. Logger còn thiếu

Hiện pipeline preprocessing tốt hơn, nhưng còn thiếu node/script chính thức:

```text
HumanStateArray -> CSV
```

Đây là blocker thực tế cho bước thu dataset.

---

## 7. Mức sẵn sàng theo subsystem

| Subsystem | Mức sẵn sàng | Nhận xét |
|---|---:|---|
| Mathematical model mapping | ~70% | Per-human logic đã đúng, còn costmap/acados/differentiability |
| Dataset preprocessing | ~75% | Schema/split/resample tốt, thiếu logger và cần sửa stale track |
| Perception | ~35% | Có node/detector/tracker nhưng stale pruning và TensorRT runtime còn lỗi |
| Prediction/LSTM | ~55% | Dataset/training tốt hơn, TensorRT engine/profile còn rủi ro |
| Context/adaptive params | ~75% | Khá đúng mô hình mới |
| Solver CasADi | ~65% | Per-human cost/constraint ổn, obstacle/costmap chưa có |
| Solver acados | ~10% | Stub |
| Controller runtime | ~45% | Có loop nhưng reference vẫn straight-line, chưa Semantic A*/P_ref |
| ROS build | ~80% | Build được |
| Real robot readiness | ~30% | Chưa nên chạy thí nghiệm chính |
| IJAT experiment readiness | ~35% | Chưa đủ |

---

## 8. Kết luận cuối

Repository **chưa xong**.

Nhưng phần đặc tả và một phần implementation đã đi đúng hướng.

Những phần đã khá tốt:

- per-human `phi_j`,
- per-human `d_safe_j`,
- human cost/constraint per-human,
- `Q_diag[3]`,
- EMA + dwell-time,
- uncertainty stale growth,
- dataset split/window/normalization.

Những phần chưa được xem là hoàn thành:

- perception tracker còn lỗi stale pruning,
- dataset logger thật chưa có,
- TensorRT runtime chưa an toàn,
- TensorRT engine dynamic profile chưa rõ,
- acados backend chưa implement,
- controller chưa nhận `P_ref` từ Semantic A*,
- obstacle/costmap cost chưa implement,
- mock fallback có thể che lỗi production.

Ưu tiên sửa tiếp theo:

1. Sửa stale track pruning trong perception.
2. Thêm dataset logger chuẩn `HumanStateArray -> CSV`.
3. Siết mock fallback để production không chạy giả.
4. Sửa TensorRT stream/synchronization.
5. Thêm TensorRT optimization profile.
6. Cho controller nhận `P_ref` thật thay vì straight line.
7. Implement obstacle/costmap soft cost.
8. Sau đó mới bắt đầu thu dataset thật.

Kết luận thực dụng:

> Hiện repo đã đủ để tiếp tục phát triển có định hướng, nhưng **chưa nên thu dataset thật chính thức** và **chưa nên dùng kết quả hiện tại để viết thực nghiệm IJAT**. Blocker đầu tiên phải xử lý là perception stale-track, vì nếu dữ liệu đầu vào sai thì toàn bộ LSTM và CCA-NMPC phía sau đều bị kéo sai theo.
