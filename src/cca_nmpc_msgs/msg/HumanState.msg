# HumanState.msg — corresponds to S_h (Eq. 5.1)
# Timestamp semantics: header.stamp = observation time from detector
std_msgs/Header header
int32 track_id
float64 x              # x_h [m]
float64 y              # y_h [m]
float64 vx             # v_x, human velocity x [m/s]
float64 vy             # v_y, human velocity y [m/s]
float64 confidence     # c, detection confidence [0,1] — MUST validate at runtime
