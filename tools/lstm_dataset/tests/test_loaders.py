"""Tests for loaders.py: identity key, metadata retention, boolean parsing."""

from __future__ import annotations

import pytest

from tools.lstm_dataset.loaders import _parse_bool, load_csv


@pytest.mark.parametrize(
    "value, expected",
    [
        ("True", True), ("true", True), ("1", True), ("yes", True), ("t", True),
        ("False", False), ("false", False), ("0", False), ("", False), ("no", False),
        (True, True), (False, False), (1, True), (0, False),
    ],
)
def test_parse_bool_handles_text_and_native(value, expected):
    # bool("False") is True in Python; the flag must not silently mislabel a
    # predicted fill as a tracker observation.
    assert _parse_bool(value) is expected


def _write_csv(path, rows_header, rows):
    path.write_text(rows_header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_load_csv_groups_by_run_id(tmp_path):
    csv = tmp_path / "t.csv"
    header = "session_id,run_id,sequence_id,timestamp,track_id,x,y,vx,vy,confidence"
    # Same session/sequence/track_id but different run_id -> two trajectories.
    rows = [
        "s,run_0,0,0.0,5,0,0,0,0,0.9",
        "s,run_0,0,0.1,5,0.1,0,1,0,0.9",
        "s,run_1,0,0.0,5,9,9,0,0,0.9",
        "s,run_1,0,0.1,5,9.1,9,1,0,0.9",
    ]
    _write_csv(csv, header, rows)
    tracks = load_csv(csv)
    assert set(tracks) == {("s", "run_0", 0, 5), ("s", "run_1", 0, 5)}


def test_load_csv_retains_optional_metadata_and_is_observed(tmp_path):
    csv = tmp_path / "t.csv"
    header = (
        "session_id,run_id,sequence_id,timestamp,track_id,x,y,vx,vy,confidence,"
        "subject_id,scenario_id,environment_id,frame_id,is_observed"
    )
    rows = [
        "s,run_0,0,0.0,1,0,0,0,0,0.9,alice,crossing,lab,map,True",
        "s,run_0,0,0.1,1,0.1,0,1,0,0.9,alice,crossing,lab,map,False",
    ]
    _write_csv(csv, header, rows)
    recs = load_csv(csv)[("s", "run_0", 0, 1)]
    assert recs[0].subject_id == "alice"
    assert recs[0].scenario_id == "crossing"
    assert recs[0].environment_id == "lab"
    assert recs[0].frame_id == "map"
    assert recs[0].is_observed is True
    assert recs[1].is_observed is False  # "False" must parse to False, not True


def test_load_csv_empty_optional_cells_use_defaults(tmp_path):
    """An optional column present but with an empty cell must fall back to the
    documented default, not the string 'nan' or a bogus True is_observed."""
    csv = tmp_path / "t.csv"
    header = (
        "session_id,run_id,sequence_id,timestamp,track_id,x,y,vx,vy,confidence,"
        "subject_id,scenario_id,environment_id,frame_id,is_observed"
    )
    # Second row leaves subject_id/scenario_id/environment_id/frame_id/is_observed empty.
    rows = [
        "s,run_0,0,0.0,1,0,0,0,0,0.9,alice,crossing,lab,map,True",
        "s,run_0,0,0.1,1,0.1,0,1,0,0.9,,,,,",
    ]
    _write_csv(csv, header, rows)
    recs = load_csv(csv)[("s", "run_0", 0, 1)]
    assert recs[1].subject_id == "unknown"
    assert recs[1].scenario_id == "unknown"
    assert recs[1].environment_id == "unknown"
    assert recs[1].frame_id == ""
    # Empty is_observed cell must NOT be silently coerced to a tracker
    # observation; conservative default is False.
    assert recs[1].is_observed is False
    # And the grouping must not create a bogus "nan" subject group.
    assert recs[0].subject_id == "alice"


def test_load_csv_missing_run_id_column_raises(tmp_path):
    csv = tmp_path / "t.csv"
    header = "session_id,sequence_id,timestamp,track_id,x,y,vx,vy,confidence"
    _write_csv(csv, header, ["s,0,0.0,1,0,0,0,0,0.9"])
    with pytest.raises(ValueError, match="run_id"):
        load_csv(csv)


def test_load_csv_empty_run_id_cell_raises(tmp_path):
    """Regression (Codex P1): an empty run_id cell would stringify to 'nan' and
    merge physically distinct runs under one trajectory key. Reject it instead."""
    csv = tmp_path / "t.csv"
    header = "session_id,run_id,sequence_id,timestamp,track_id,x,y,vx,vy,confidence"
    # Row 2 leaves run_id empty -> pandas NaN. Without validation it would join
    # the same (session, sequence, track) as run_0 rows, corrupting isolation.
    rows = [
        "s,run_0,0,0.0,5,0,0,0,0,0.9",
        "s,,0,0.1,5,0.1,0,1,0,0.9",
    ]
    _write_csv(csv, header, rows)
    with pytest.raises(ValueError, match="run_id"):
        load_csv(csv)
