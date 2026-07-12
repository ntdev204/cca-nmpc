---
name: codex-review-loop
description: How to invoke the codex CLI non-interactively to review changes and drive the accept loop for this project
---

# Codex Review Loop

The project goal requires every implementation to be reviewed and **accepted by codex** before it's considered done. codex CLI (`codex-cli 0.125.0`) is on PATH.

## Invocation (learned the hard way)

`codex exec review` **cannot** take a prompt argument together with `--uncommitted` / `--base` / `--commit`. The CLI errors: `the argument '--uncommitted' cannot be used with '[PROMPT]'`. Even piping via stdin `-` counts as a positional and fails.

**Working forms:**

- Review all staged+unstaged+untracked changes (no custom prompt):
  ```bash
  codex exec review --uncommitted
  ```
- Review vs a base branch:
  ```bash
  codex exec review --base develop
  ```
- Custom-prompt review of the whole repo state (no diff flag):
  ```bash
  echo "review instructions..." | codex exec review -
  ```
  or `codex exec review "instructions"` — but this reviews working tree without a scoped diff.

To get BOTH a scoped diff AND custom context: commit the change first, then `codex exec review --commit <sha>` (still no prompt), OR put context in the code/commit message so the default review picks it up.

## Practical notes

- Runs long (30-90s+). Launch with `run_in_background: true` or expect a background task ID; read the `.output` file when done.
- Output ends with a `codex` section: a one-line verdict then `Review comment:` bullets tagged `[P1]`/`[P2]`/`[P3]` with `file:line`. P1≈CRITICAL, P2≈HIGH/MEDIUM, P3≈LOW.
- "changes appear consistent" / no P1 bullets == acceptance for our purposes. Fix P1/P2 before declaring a package done; note P3s.
- Harmless stderr noise: `ERROR codex_core::session: failed to record rollout items: thread ... not found`. Ignore it.

## Accept-loop procedure

1. Implement + unit-test on Windows (`pytest` from package root — see [[env-toolchain]]).
2. `codex exec review --uncommitted`.
3. Address every P1/P2. Re-review until no P1/P2 remain.
4. Then verify in WSL colcon (see [[wsl-sync-workflow]]).
