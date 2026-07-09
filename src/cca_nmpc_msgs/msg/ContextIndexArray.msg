# ContextIndexArray.msg
# Timestamp semantics: header.stamp = context computation time
std_msgs/Header header
ContextIndex[] contexts
float64 phi_aggregate         # max_j(phi_j), raw value, Eq. 9.1
float64 phi_aggregate_used    # max_j(phi_j_used), gated value used by adaptive params
float64 d_h_aggregate         # min_j(d_j), Eq. 9.2
