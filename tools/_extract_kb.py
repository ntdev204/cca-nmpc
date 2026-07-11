import subprocess
import os

out_path = r'D:/Research/cca-nmpc/tools/_orig_kb.txt'
os.makedirs(os.path.dirname(out_path), exist_ok=True)

r = subprocess.run(
    ['git', '-C', r'D:/Research/cca-nmpc', 'show',
     'HEAD:src/rai_robot_keyboard/rai_robot_keyboard/rai_keyboard.py'],
    capture_output=True,
)

with open(out_path, 'wb') as f:
    f.write(r.stdout)

meta_path = r'D:/Research/cca-nmpc/tools/_orig_kb_meta.txt'
with open(meta_path, 'w', encoding='utf-8') as f:
    f.write(f'rc={r.returncode}\n')
    f.write(f'size={len(r.stdout)}\n')
    f.write(f'lines={r.stdout.count(b"\n")}\n')
    f.write(f'stderr={r.stderr.decode("utf-8", errors="replace")[:500]}\n')
