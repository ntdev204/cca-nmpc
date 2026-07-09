# AdaptiveParams.msg — corresponds to Section 10 of Mathematical Model
# Timestamp semantics: header.stamp = parameter computation time
std_msgs/Header header
float64 d_safe_aggregate     # for monitoring/diagnostics only; per-human d_safe used internally by solver
float64 vx_max               # Eq. 10.2
float64 vy_max               # Eq. 10.2
float64 omega_max            # Eq. 10.2
float64[3] Q_diag            # diagonal of Q(phi), Eq. 10.3 — state [x_r, y_r, theta_r]
float64 phi_aggregate_used   # phi value this parameter set was generated from
