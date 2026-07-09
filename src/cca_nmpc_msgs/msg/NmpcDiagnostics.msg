# NmpcDiagnostics.msg — Section 12.1 (feasibility) and Section 9 of solver_design (fallback)

# Diagnostic level constants (mirrors diagnostic_msgs/DiagnosticStatus)
uint8 LEVEL_OK=0
uint8 LEVEL_WARN=1
uint8 LEVEL_ERROR=2
uint8 LEVEL_STALE=3

# Fallback reason constants
uint8 FALLBACK_NONE=0
uint8 FALLBACK_SOLVER_FAILED=1
uint8 FALLBACK_TIMEOUT=2
uint8 FALLBACK_HELD_PREVIOUS=3
uint8 FALLBACK_SAFE_STOP=4

# Timestamp semantics: header.stamp = NMPC solve completion time
std_msgs/Header header
uint8 diagnostic_level         # see LEVEL_* constants above
bool solver_success
float64 solve_time_ms
SlackValue[] slacks            # s_j per tracked human, Eq. 12.3 — each entry has track_id + slack
int32 num_humans_active
float64 cost_total             # J, Eq. 11.1
float64 cost_track
float64 cost_control
float64 cost_smooth
float64 cost_human
float64 cost_obstacle
float64 cost_terminal
bool fallback_triggered        # true if this cycle published held/decelerating control or safe stop
uint8 fallback_code            # see FALLBACK_* constants above
string fallback_reason         # human-readable debug string, optional
