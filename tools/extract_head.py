#!/usr/bin/env python3
import subprocess
import os

os.chdir(r'D:/Research/cca-nmpc')
result = subprocess.run(
    ['git', 'show', 'HEAD:src/rai_robot_keyboard/rai_robot_keyboard/rai_keyboard.py'],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    with open(r'D:/Research/cca-nmpc/tools/_orig_kb.txt', 'w', encoding='utf-8') as f:
        f.write(result.stdout)
    print(f"SUCCESS: Wrote {len(result.stdout)} bytes, {result.stdout.count(chr(10))} lines")
    lines = result.stdout.split('\n')
    print("First 45 lines:")
    for line in lines[:45]:
        print(line)
else:
    print(f"ERROR: {result.stderr}")
