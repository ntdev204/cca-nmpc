# Windows pytest gotchas (this project)

Learned while building the offline Python packages on Windows Python 3.14.

## np.load handle locks the file (WinError 32)

`np.load("x.npz")` returns a lazy `NpzFile` that keeps the file OPEN. On Windows a `tempfile.TemporaryDirectory()` cleanup then fails with `PermissionError: [WinError 32] The process cannot access the file because it is being used by another process`.

Fix: use it as a context manager and copy out what you need:
```python
with np.load(npz_path) as data:
    inputs = data["inputs"].copy()
    targets = data["targets"].copy()
```
Applies anywhere a test writes .npz into a temp dir then loads it (dataset, training, harness).

## Trajectory-level split needs enough trajectories

`math.ceil(n*0.7)` with n=3 tracks sends ALL to train, leaving val/test empty. Integration tests must generate >= ~7-12 distinct track_ids to exercise a real 70/15/15 split and assert each partition is non-empty.

## Running package tests

Tests import `tools.lstm_dataset.*` / `cca_nmpc_perception.*` etc. Run pytest from the REPO ROOT for `tools.*` (they use the full package path), but perception tests import `cca_nmpc_perception.*` so run those from the package root `src/cca_nmpc_perception/`. Check the test's import style first.
