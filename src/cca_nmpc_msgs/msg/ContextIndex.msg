# ContextIndex.msg — corresponds to per-human phi_j, phi_j_filtered, phi_j_used (Eqs. 8.1, 9.1, 12.6, 12.7)
int32 track_id
float64 z               # raw context score, Eq. 8.2
float64 phi_j           # sigma(z), Eq. 8.1
float64 phi_j_filtered  # EMA-smoothed value, Eq. 12.6
float64 phi_j_used      # dwell-time-gated value (post-EMA), Eq. 12.7
float64 d_j             # current distance to this human, Eq. 7.2
