# HumanSafetyDistance.msg — per-human adaptive safety distance d_safe(phi_j), Eq. 10.1
int32 track_id
float64 d_safe         # d_safe(phi_j) for this human, used by the solver's per-human constraint (Eq. 12.3)
