Kết luận review
Repository chưa hoàn thành và chưa an toàn để:
thu dataset chính thức;
chạy robot thật;
benchmark CCA-NMPC;
dùng kết quả cho bài báo IJAT.
Các mô-đun toán học thuần Python đã có nền tảng khá tốt, nhưng implementation chạy thực tế còn nhiều sai lệch quan trọng so với mô hình toán và docs. Đặc biệt, ràng buộc an toàn người hiện không được áp dụng trên toàn horizon.
Findings nghiêm trọng
P1 — Ràng buộc người chỉ áp dụng ở terminal state
Trong [casadi_solver.py (line 124)](/D:/Research/cca-nmpc/src/cca_nmpc_control/cca_nmpc_control/nmpc_solver/casadi_solver.py:124), ràng buộc
\[
d*{j,k}+s_j\ge d*{\mathrm{safe},j}
\]chỉ được thêm tại \(X*N\), không phải mọi \(X_k\). \_n_ineq cũng chỉ bằng \(M\), thay vì \(M(N+1)\).
Tôi chạy một trường hợp kiểm chứng:
required d_safe: 1.0 m
minimum predicted distance: ~0.0 m
reported slack: 0.000006
solver_success: True
Robot vi phạm gần như toàn bộ vùng an toàn nhưng solver vẫn báo thành công vì người ở xa tại terminal state.
Cần:
đặt constraint trong vòng lặp horizon cho mọi \(k,j\);
đặt thêm terminal constraint nếu cần;
sửa số lượng inequality;
quyết định dùng \(s_j\) chung hoặc \(s*{j,k}\);
thêm test kiểm tra khoảng cách tại từng stage, không chỉ solver*success.
P1 — Sai trục thời gian giữa LSTM và NMPC
LSTM chạy 8 Hz với \(H=12\), tức dự báo tại khoảng 0.125 s/bước. NMPC chạy 20 Hz với \(dt=0.05\) s.
Nhưng [casadi_solver.py (line 258)](/D:/Research/cca-nmpc/src/cca_nmpc_control/cca_nmpc_control/nmpc_solver/casadi_solver.py:258) đưa trực tiếp phần tử thứ \(i\) của LSTM vào stage \(i\) của NMPC. Khi thiếu, \_fit_to_len() chỉ lặp lại phần tử cuối.
Hậu quả:
dự báo \(t+0.125\) bị dùng như \(t+0\);
dự báo \(t+1.5\) có thể bị dùng tại stage robot \(t+0.55\);
phần cuối horizon giữ người đứng yên tại dự báo cuối.
Đây là lỗi time-alignment có thể làm ràng buộc né người sai vị trí.
Cần nội suy:
\[
\hat P_h(t_k+iT*{\mathrm{NMPC}})
\]từ timestamp và sampling period thật của LSTM, đồng thời thêm vị trí người hiện tại tại stage 0.
P1 — Không có tránh vật cản tĩnh
Docs yêu cầu \(J*{\mathrm{obstacle}}\), nhưng code trả cố định:
"obstacle_cost": 0.0
tại [casadi_solver.py (line 350)](/D:/Research/cca-nmpc/src/cca_nmpc_control/cca_nmpc_control/nmpc_solver/casadi_solver.py:350).
nmpc_controller_node cũng không subscribe /local_costmap/costmap, dù topic được đặc tả tại [05_ros_topics.md (line 37)](/D:/Research/cca-nmpc/docs/05_ros_topics.md:37).
Reference hiện chỉ là đường thẳng goal–robot tại [nmpc_controller_node.py (line 181)](/D:/Research/cca-nmpc/src/cca_nmpc_control/cca_nmpc_control/nmpc_controller_node.py:181), không phải \(P*{\mathrm{ref}}\) từ Semantic A\*/planner. Vì vậy robot có thể đi xuyên tường hoặc kệ.
Ngoài ra docs tự xung đột:
01_mathematical_model.md vẫn liệt kê obstacle như hard constraint;
08_solver_design.md quyết định chỉ dùng soft cost.
Cần chốt một formulation và triển khai cost có đạo hàm, ví dụ signed-distance field hoặc costmap nội suy trơn. Không đưa raw grid lookup không khả vi trực tiếp vào SQP/IPOPT.
P1 — Perception node không khởi động trên ROS2 Humble
Smoke test thực tế thất bại tại [perception_node.py (line 10)](/D:/Research/cca-nmpc/src/cca_nmpc_perception/cca_nmpc_perception/perception_node.py:10):
ImportError: cannot import name 'SensorDataQoS' from 'rclpy.qos'
Python ROS2 Humble dùng qos_profile_sensor_data, không phải SensorDataQoS.
Ngay cả sau khi sửa import, cấu hình mặc định vẫn tự mâu thuẫn:
require_depth_alignment: true;
depth_image_topic: /camera/depth/image_raw;
code yêu cầu tên topic chứa aligned, registered hoặc depth_to_color.
Do đó [perception_node.py (line 119)](/D:/Research/cca-nmpc/src/cca_nmpc_perception/cca_nmpc_perception/perception_node.py:119) sẽ tiếp tục từ chối cấu hình mặc định.
P1 — Kalman tracker dự đoán quá xa khi mất detection
[track_manager.py (line 69)](/D:/Research/cca-nmpc/src/cca_nmpc_perception/cca_nmpc_perception/track_manager.py:69) dự đoán theo:
dt = current_time - last_update_time
nhưng prediction không cập nhật thời điểm trạng thái. Với nhiều frame mất detection, mỗi lần lại tích phân từ timestamp đo cuối trên một trạng thái đã được dự đoán trước đó.
Kiểm chứng:
expected x at t3 under constant velocity: 1.995 m
tracker output: 2.492 m
Cần tách:
state_time: thời điểm trạng thái đã được propagate;
last_measurement_time: dùng để prune và tính observation age.
Hiện track bị mất detection vẫn được publish lại với timestamp mới và confidence cũ, khiến context/LSTM tưởng đó là phép đo mới.
P1 — Uncertainty stale growth chỉ tồn tại trong helper, không chạy online
UncertaintyEstimator.age() được định nghĩa tại [uncertainty.py (line 66)](/D:/Research/cca-nmpc/src/cca_nmpc_prediction/cca_nmpc_prediction/uncertainty.py:66), nhưng không có code production nào gọi nó.
Ngoài ra [prediction_node.py (line 97)](/D:/Research/cca-nmpc/src/cca_nmpc_prediction/cca_nmpc_prediction/prediction_node.py:97) có thể so cùng một one-step prediction với nhiều state callback trước lần LSTM kế tiếp, làm rolling variance sai time index.
context_node còn dùng:
sigma_tilde = uncertainties.get(track_id, 0.0)
tại [context_node.py (line 151)](/D:/Research/cca-nmpc/src/cca_nmpc_context/cca_nmpc_context/context_node.py:151). Thiếu uncertainty lại được hiểu là “hoàn toàn chắc chắn”, trái triết lý fail-safe.
Cần:
khớp prediction với đúng target timestamp;
refresh error đúng một lần cho mỗi prediction;
gọi stale-growth theo elapsed time;
khi uncertainty thiếu/stale, dùng giá trị bảo thủ hoặc kích hoạt dropout.
P1 — Dataset có thể nối nhầm nhiều trajectory và gây leakage
[loaders.py (line 41)](/D:/Research/cca-nmpc/tools/lstm_dataset/loaders.py:41) và [split.py (line 54)](/D:/Research/cca-nmpc/tools/lstm_dataset/split.py:54) dùng track_id làm định danh trajectory toàn cục.
Nhưng track_id thường reset khi:
restart tracker;
ghi rosbag mới;
đổi session;
mất dấu rồi sinh ID lại.
Nếu ghép nhiều buổi quay, các track_id=1 khác nhau sẽ bị nối thành một trajectory. Split theo track_id cũng không bảo đảm tách người, session, môi trường hoặc scenario.
Schema cần tối thiểu:
dataset_version
session_id
run_id
sequence_id
track_id
subject_id (nếu có)
scenario_id
environment_id
timestamp
frame_id
x,y,vx,vy,confidence
is_observed
Khóa trajectory nên là (session_id, sequence_id, track_id) và split theo session/person/environment tùy mục tiêu generalization.
P1 — Resampling không tạo đúng dt và tạo dữ liệu giả qua khoảng mất track
[resample.py (line 52)](/D:/Research/cca-nmpc/tools/lstm_dataset/resample.py:52) dùng:
np.linspace(t_min, t_max, ceil(span/dt)+1)
Kiểm chứng với dt=0.06:
timestamps: [0.00, 0.05, 0.10]
actual dt: 0.05
Vì vậy horizon vật lý và đạo hàm vận tốc sai.
Ngoài ra code nội suy tuyến tính xuyên qua khoảng mất track. [windowing.py (line 64)](/D:/Research/cca-nmpc/tools/lstm_dataset/windowing.py:64) còn cho phép tới 30% window nằm trong gap. Đây là synthetic interpolation, không còn là chuyển động thật.
Cần:
dùng grid chính xác theo np.arange;
split trajectory tại gap lớn;
loại toàn bộ window cắt qua gap lớn;
không dùng nội suy qua occlusion dài làm target.
P1 — Rosbag loader sai message và chưa có logger dataset
[loaders.py (line 118)](/D:/Research/cca-nmpc/tools/lstm_dataset/loaders.py:118) đọc msg.states, trong khi message thực là:
HumanState[] humans
Loader cũng dùng bag write-time thay vì msg.header.stamp.
Repository không có node/logger chuyển /human_states thành CSV hoặc rosbag profile như docs tuyên bố. Vì vậy hiện chưa có đường thu Dataset 01 hoàn chỉnh.
Trước khi quay dữ liệu thật cần có:
rosbag record profile: RGB, depth, camera info, TF, odom, /human_states;
CSV exporter dùng observation timestamp và frame;
metadata camera/calibration/config/git SHA;
validator phát hiện duplicate timestamp, ID switch, teleport, missing frame và invalid confidence.
P1 — TensorRT paths chưa chạy đúng trên Jetson
Cả YOLO và LSTM đều tạo stream tạm rồi copy output đồng bộ ngay:
[detector.py (line 155)](/D:/Research/cca-nmpc/src/cca_nmpc_perception/cca_nmpc_perception/detector.py:155)
[lstm_infer.py (line 142)](/D:/Research/cca-nmpc/src/cca_nmpc_prediction/cca_nmpc_prediction/lstm_infer.py:142)
Stream không được giữ và không được synchronize trước memcpy_dtoh. Code cũng không quản lý CUDA context rõ ràng.
ONNX có dynamic batch, nhưng [build_engine.py (line 47)](/D:/Research/cca-nmpc/tools/lstm_training/build_engine.py:47) không tạo TensorRT optimization profile, nên engine build có thể thất bại.
Nguy hiểm hơn, khi TensorRT/model thiếu, prediction tự động chạy MockLSTMPredictor tại [prediction_node.py (line 88)](/D:/Research/cca-nmpc/src/cca_nmpc_prediction/cca_nmpc_prediction/prediction_node.py:88). Smoke test xác nhận node báo warning rồi vẫn chạy. Một thử nghiệm tưởng đang dùng LSTM có thể thực chất đang dùng Constant Velocity.
Production mode phải fail-fast; mock chỉ được bật bằng tham số explicit và diagnostics phải công bố backend đang hoạt động.
P1 — Control không kiểm tra freshness và chọn sai người khi quá số slot
[nmpc_controller_node.py (line 191)](/D:/Research/cca-nmpc/src/cca_nmpc_control/cca_nmpc_control/nmpc_controller_node.py:191) lấy predictions[:max_humans], tức chọn theo thứ tự message, không phải người gần/nguy hiểm nhất.
Khi không có prediction, node vẫn giải NMPC với zero human slots. Khi prediction/context/adaptive params cũ, chúng được giữ vô thời hạn. Không có kiểm tra:
prediction_stamp;
AdaptiveParams age;
context age;
odometry age;
frame ID.
Trong môi trường đông người, người gần nhất có thể bị loại khỏi solver.
Cần chọn top-\(M\) theo risk, ví dụ TTC/\(\phi_j\)/minimum predicted distance, và safe-stop hoặc conservative fallback khi input hết hạn.
Findings mức P2
Pseudo-ground-truth: /human_states là output của YOLO–depth–Kalman, không phải ground truth. ADE/FDE trên future output của cùng tracker chỉ đánh giá khả năng dự đoán pseudo-label. Cần gọi đúng tên và có một subset tham chiếu độc lập như mocap, overhead annotation hoặc khảo sát mốc sàn.

Absolute map coordinates: model học trực tiếp [x,y,vx,vy]. Điều này dễ ghi nhớ map origin và khó trộn ETH/UCY với dữ liệu robot. Nên dùng tọa độ tương đối so với observation cuối, có thể quay theo heading, rồi transform prediction về map frame.

Input toán học không nhất quán: \(S_h\) trong [01_mathematical_model.md (line 111)](/D:/Research/cca-nmpc/docs/01_mathematical_model.md:111) gồm confidence \(c\), nhưng model và tensor chỉ có 4 channels. Cần chốt LSTM input là 4 hay 5; nếu confidence chỉ dùng context thì sửa Eq. 6.1.

Calibration chưa đủ giá trị khoa học: [sensitivity.py (line 48)](/D:/Research/cca-nmpc/tools/context_calibration/sensitivity.py:48) chỉ chạy một scenario tại mean \(\phi\), trong khi slack làm solver hầu như luôn feasible. “Solve-success degradation” vì vậy gần như luôn 0%. Calibration cũng báo train accuracy, không có held-out split, và logistic weights không bị ràng buộc dấu nên có thể học \(w_d<0\) hoặc \(w_u<0\), trái safety monotonicity.

Warm start chưa được dùng: helper shift-and-append có test, nhưng [casadi_solver.py (line 275)](/D:/Research/cca-nmpc/src/cca_nmpc_control/cca_nmpc_control/nmpc_solver/casadi_solver.py:275) chỉ tái sử dụng nguyên vector nghiệm cũ.

Timeout chỉ phát hiện sau khi solver trả về: [nmpc_controller_node.py (line 153)](/D:/Research/cca-nmpc/src/cca_nmpc_control/cca_nmpc_control/nmpc_controller_node.py:153) gọi blocking solve(), rồi mới so thời gian. Đây không phải deadline thực; nếu IPOPT treo 500 ms, control loop cũng bị chặn 500 ms.

acados chưa được triển khai: [acados_solver.py (line 34)](/D:/Research/cca-nmpc/src/cca_nmpc_control/cca_nmpc_control/nmpc_solver/acados_solver.py:34) toàn bộ là NotImplementedError.

Dependency chưa đóng gói: cca_nmpc_control không khai báo CasADi. colcon test và smoke run đều thất bại với ModuleNotFoundError: casadi. Các package Python cũng chỉ khai báo setuptools, không khóa NumPy, pandas, OpenCV, TensorRT/PyCUDA.

Docs drift: docs/07 dùng model_path ONNX nhưng runtime dùng engine_path; docs dùng Q_diag trong khi message thật là q_diag; docs 08 nói solver chưa implement dù CasADi đã tồn tại; /goal_pose có trong code nhưng không có trong topic spec.

Không có combined bringup: src/cca_nmpc_bringup hiện chỉ có resource marker, không có package.xml, setup.py, config hay launch tổng.

Traceability chính
Requirement Implementation Trạng thái
Mecanum \(v*x,v_y,\omega\) + RK4 ocp_spec.py ✅
Per-human \(\phi_j\) context loop ✅
EMA → dwell-time PhiSmoother ✅
\(d*{\mathrm{safe}}(\phi*j)\) upstream adaptive → message → solver ✅
\(Q\) diagonal 3 phần tử q_diag[3] ✅
Velocity caps có floor saturated_velocity_limits ✅
Per-human \(J*{\mathrm{human}}\) solver stage cost ✅
Per-human safety constraint toàn horizon terminal-only ❌
Fixed human slots + dummy implemented ⚠️ chọn top-\(M\) sai
Slack implemented per human ⚠️ constraint sai horizon
Stale uncertainty growth helper only ❌
\(J\_{\mathrm{obstacle}}\) luôn 0 ❌
Differentiable obstacle representation chưa có ❌
Shift warm-start helper không được gọi ❌
acados/SQP-RTI stub ❌
Dataset schema cơ bản có ⚠️ thiếu session/sequence
Fixed-rate resampling sai dt ❌
Leakage-safe split chỉ theo track_id ⚠️
Train-only normalization đã sửa đúng ✅
Dataset logger không có ❌
Independent ground truth không có ❌
LSTM training/export scaffold có, synthetic tests ⚠️
Real model/data artifacts không có ❌
End-to-end ROS bringup không có ❌

Kết quả kiểm chứng
Pure Python pytest: 195 passed.
ROS2 colcon build: 6/6 package build thành công.
ROS2 colcon test: messages, perception, prediction, context và adaptive pass; control fail do thiếu CasADi.
Ruff: 58 lỗi.
git diff --check: pass, chỉ có cảnh báo LF→CRLF.
Smoke run:perception: fail import QoS;
control: fail thiếu CasADi;
prediction: chạy nhưng tự động dùng mock.

codex exec review --uncommitted đã được chạy theo skill. Lần đầu model cấu hình không tương thích; lần chạy lại với gpt-5.4 bị lỗi Windows sandbox nên không thể tạo secondary review. Theo quy tắc review-loop, trạng thái hiện tại không thể được coi là accepted, và các P1/P2 trên cũng chưa được xử lý.
Mức sẵn sàng
Subsystem Sẵn sàng
Messages 90%
Math helpers: context/adaptive 75%
Perception core 40%
Perception runtime 10%
Dataset preprocessing 40%
Dataset collection 0–5%
LSTM training scaffold 55%
Real trained predictor 0%
Prediction runtime 25%
CasADi OCP 35%
acados deployment 0%
Control node 25%
Bringup/end-to-end robot 5%
Experimental/paper readiness 0–10%
