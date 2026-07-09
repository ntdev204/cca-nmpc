# HumanPrediction.msg — corresponds to S_h_hat (Eq. 6.2), one predicted trajectory per tracked human
# All arrays MUST have length H (prediction horizon)
int32 track_id
int32 horizon_length   # H, number of prediction steps (for validation)
float64[] x_hat        # x_h_hat over horizon H
float64[] y_hat        # y_h_hat over horizon H
float64[] vx_hat       # v_x_hat over horizon H
float64[] vy_hat       # v_y_hat over horizon H
builtin_interfaces/Time prediction_stamp   # time prediction was generated (staleness tracking, Section 13.1)
