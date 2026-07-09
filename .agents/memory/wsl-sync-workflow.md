---
name: wsl-sync-workflow
description: Git sync procedure between Windows dev copy and WSL build/test copy for cca-nmpc
metadata:
  type: project
---

Code is written on Windows at `D:\Research\cca-nmpc`; built/tested in WSL at `~/cca-nmpc`. **Rule: no code changes in WSL** — WSL only builds/tests.

**Windows git remotes:** the GitHub remote is named `cca-nmpc` (NOT `origin`). Push to develop: `git push cca-nmpc HEAD:develop`.

**WSL git remotes:** remote is `origin` → same GitHub repo (https://github.com/ntdev204/cca-nmpc.git). Pull: `git pull origin develop`.

**Full loop from Windows:**
1. Edit + unit-test on Windows (`pytest` from package root).
2. `git add <files> && git commit -m "..."` then `git push cca-nmpc HEAD:develop`.
3. `wsl bash -c "cd ~/cca-nmpc && git pull origin develop && source /opt/ros/humble/setup.bash && colcon build --symlink-install --packages-select <pkg> && colcon test --packages-select <pkg> --event-handlers console_direct+"`

**Gotchas learned:**
- CRLF↔LF: Windows files show as "modified" in WSL from line endings only. `git checkout -- <dir>` to reset noise.
- `git checkout ~/cca-nmpc/.git/index.lock` stale lock: `rm -f ~/cca-nmpc/.git/index.lock`.
- Untracked files in WSL can block merge — remove them if they're stale copies.
- `package.xml` format3 XSD requires element order: `test_depend` BEFORE `member_of_group`. xmllint CTest enforces it. See [[env-toolchain]].
- `astra_camera` build fails in WSL (missing `camera_info_manager`) — pre-existing hardware-driver issue, unrelated to our packages. Build our packages with `--packages-select`.
