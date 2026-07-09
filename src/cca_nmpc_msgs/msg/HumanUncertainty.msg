# HumanUncertainty.msg — corresponds to sigma_h / sigma_h_tilde (Eq. 6.3, 8.3)
int32 track_id
float64 sigma_h         # rolling prediction-error variance, Eq. 6.3
float64 sigma_h_clipped # sigma_h_tilde, clipped to [0,1], Eq. 8.3
