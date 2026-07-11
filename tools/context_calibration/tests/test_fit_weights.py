"""Tests for stage-1 weight fit + feature reconstruction (CB-02, CB-03)."""
import csv


from tools.context_calibration.load import load_calibration_csv, danger_label
from tools.context_calibration.features import compute_features
from tools.context_calibration.fit_weights import (
    fit_weights, classification_accuracy,
)


def _write_csv(path, rows):
    cols = [
        "robot_x", "robot_y", "robot_theta", "robot_vx", "robot_vy",
        "human_x", "human_y", "human_vx", "human_vy", "confidence",
        "sigma_h_tilde", "min_future_distance", "evasive_reaction_flag",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _row(human_x, human_y, min_future, evasive):
    return {
        "robot_x": 0.0, "robot_y": 0.0, "robot_theta": 0.0,
        "robot_vx": 0.0, "robot_vy": 0.0,
        "human_x": human_x, "human_y": human_y,
        "human_vx": 0.0, "human_vy": 0.0, "confidence": 0.9,
        "sigma_h_tilde": 0.1, "min_future_distance": min_future,
        "evasive_reaction_flag": evasive,
    }


def test_loader_reads_schema(tmp_path):
    p = tmp_path / "c.csv"
    _write_csv(p, [_row(1.0, 0.0, 0.5, "true")])
    recs = load_calibration_csv(p)
    assert len(recs) == 1
    assert recs[0].evasive_reaction_flag is True


def test_loader_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    with open(p, "w", newline="") as f:
        f.write("robot_x,robot_y\n0,0\n")
    try:
        load_calibration_csv(p)
        assert False
    except ValueError:
        pass


def test_danger_label_logic(tmp_path):
    close = _row(0.3, 0.0, 0.4, "false")   # close future distance
    far_safe = _row(5.0, 0.0, 5.0, "false")
    evasive = _row(5.0, 0.0, 5.0, "true")
    p = tmp_path / "c.csv"
    _write_csv(p, [close, far_safe, evasive])
    recs = load_calibration_csv(p)
    assert danger_label(recs[0]) == 1.0   # close
    assert danger_label(recs[1]) == 0.0   # far + no evasion
    assert danger_label(recs[2]) == 1.0   # evasion


def test_fit_recovers_separating_boundary(tmp_path):
    # dangerous when human is close (small min_future_distance); safe when far.
    rows = []
    for i in range(40):
        rows.append(_row(0.4, 0.0, 0.3, "false"))   # close/danger
    for i in range(40):
        rows.append(_row(6.0, 0.0, 6.0, "false"))   # far/safe
    p = tmp_path / "c.csv"
    _write_csv(p, rows)
    recs = load_calibration_csv(p)
    feats = [compute_features(r, d0=3.0, v_max_ref=1.5) for r in recs]
    labels = [danger_label(r) for r in recs]
    weights = fit_weights(feats, labels, epochs=3000)
    acc = classification_accuracy(feats, labels, weights)
    assert acc >= 0.9


def test_features_reuse_context_math(tmp_path):
    # feature dist_term must equal (d0 - d_h)/d0 with d_h from the context core
    rec = load_calibration_csv(_single(tmp_path))[0]
    f = compute_features(rec, d0=3.0, v_max_ref=1.5)
    # human at (3,4) from origin -> d_h=5 -> (3-5)/3 = -0.6667
    assert abs(f.dist_term - ((3.0 - 5.0) / 3.0)) < 1e-6


def _single(tmp_path):
    p = tmp_path / "one.csv"
    _write_csv(p, [_row(3.0, 4.0, 2.0, "false")])
    return p
