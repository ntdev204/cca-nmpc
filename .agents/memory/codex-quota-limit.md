---
name: codex-quota-limit
description: codex CLI has a daily quota that blocks reviews when exhausted; how to work around it
metadata:
  type: project
---

The codex CLI account hits hard limits during heavy use:
- `API Error: Request rejected (429) · Daily cost limit would be exceeded. Resets at <UTC midnight>`
- `ERROR: limit exceeded, 额度用完了` (quota used up) mid-review.

When this happens codex review/exec CANNOT run until the quota resets (~00:00 UTC).

**Why:** The goal requires codex to review+accept the implementation, but codex availability is externally gated.

**How to apply:** Do NOT block on codex. Keep implementing and unit-testing on Windows (pytest — no external quota). Stage/commit verified work. Batch codex reviews (`codex exec review --uncommitted`) when quota is available (retry after reset). Track which commits still need a codex pass so none ship unreviewed. The Stop-hook goal is only met once codex has actually accepted — so a codex review pass MUST happen before final done; it can be deferred, not skipped. See [[implementation-plan-state]] and skill codex-review-loop.
