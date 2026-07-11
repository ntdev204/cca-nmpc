#!/bin/bash
set -x

# Phase 1: Cleanup
rm -f tools/_git_probe.sh tools/_git_node.js tools/_release.ps1 tools/_out.txt tools/_ps_out.txt tools/_orig_kb.txt tools/_git_out.txt

# Phase 2: Remove worlds if exists
if [ -f worlds/empty_indoor.world ]; then
  git rm -f worlds/empty_indoor.world
  rmdir worlds 2>/dev/null || true
fi

# Phase 3: Stage all changes
git add -A

# Phase 4: Unstage specific files
git reset HEAD tools/_rel.txt wsl.txt 2>/dev/null || true

# Phase 5: Check status
echo "=== GIT STATUS ==="
git status --short

# Phase 6: Commit on develop
git commit -m "refactor: strip code comments; move worlds into turn_on_rai_robot; doc alignment

- Remove comments/docstrings from cca_nmpc_* and rai_robot_* source (keep shebangs, license headers, noqa)
- Move empty_indoor.world into turn_on_rai_robot package share install
- Fix fragile CMake ../../worlds path
- Align J_human per-human notation in solver design docs
- Add tools/build_dev.sh and tools/build_sim.sh"

# Phase 7: Show recent commits
echo "=== RECENT COMMITS ==="
git log --oneline -3

# Phase 8: Merge to main
echo "=== MERGING TO MAIN ==="
git checkout main
git merge develop -m "Merge branch 'develop' into main"

# Phase 9: Cleanup branches
echo "=== CLEANING BRANCHES ==="
git worktree remove --force "D:/Research/cca-nmpc/.claude/worktrees/agent-a25f2dbaf41d2b9be" 2>/dev/null || true
for b in $(git branch | sed 's/^\*//' | tr -d ' '); do
  if [ "$b" != "develop" ] && [ "$b" != "main" ]; then
    git branch -D "$b" 2>/dev/null || true
  fi
done
git branch

# Phase 10: Push
echo "=== PUSHING ==="
git push cca-nmpc main
git push cca-nmpc develop

# Phase 11: Tag
echo "=== TAGGING ==="
git tag -a v0.2.0 -m "v0.2.0: comment cleanup, sim worlds packaging, doc alignment"
git push cca-nmpc v0.2.0

# Phase 12: Release
echo "=== CREATING RELEASE ==="
gh release create v0.2.0 --target main --title "v0.2.0" --notes "Strip inline comments/docstrings from project source (license headers, shebangs, noqa preserved); move Gazebo world into turn_on_rai_robot package; fix CMake fragile worlds path; align solver-design J_human per-human notation; add targeted build scripts. See commits since v0.1.0 on main."

# Phase 13: Back to develop
echo "=== BACK TO DEVELOP ==="
git checkout develop

# Phase 14: Final status
echo "=== FINAL STATUS ==="
git tag -l
gh release list
git status

echo "=== DONE ==="
