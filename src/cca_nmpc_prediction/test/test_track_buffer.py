"""Tests for per-track rolling buffers (Eq. 6.1)."""
import numpy as np

from cca_nmpc_prediction.track_buffer import TrackBufferManager


def _s(x): return np.array([x, 0.0, 0.0, 0.0])


def test_fills_to_L_then_ready():
    m = TrackBufferManager(L=3, max_age_sec=1.0)
    assert not m.is_ready(1)
    m.update(1, _s(0), 0.0)
    m.update(1, _s(1), 0.1)
    assert not m.is_ready(1)
    m.update(1, _s(2), 0.2)
    assert m.is_ready(1)
    assert m.get_window(1).shape == (3, 4)


def test_matches_by_track_id_not_index():
    m = TrackBufferManager(L=2, max_age_sec=1.0)
    m.update(5, _s(0), 0.0)
    m.update(9, _s(10), 0.0)
    m.update(5, _s(1), 0.1)
    m.update(9, _s(11), 0.1)
    assert m.get_window(5)[-1][0] == 1
    assert m.get_window(9)[-1][0] == 11


def test_evicts_stale():
    m = TrackBufferManager(L=2, max_age_sec=0.5)
    m.update(1, _s(0), 0.0)
    m.update(1, _s(1), 0.1)
    m.prune(current_time=1.0)  # 0.9s > 0.5s max age
    assert 1 not in m.active_tracks()


def test_ready_tracks_list():
    m = TrackBufferManager(L=1, max_age_sec=1.0)
    m.update(1, _s(0), 0.0)
    m.update(2, _s(0), 0.0)
    assert set(m.ready_tracks()) == {1, 2}


def test_window_rejects_wrong_shape():
    m = TrackBufferManager(L=2, max_age_sec=1.0)
    try:
        m.update(1, np.zeros(3), 0.0)
        assert False
    except ValueError:
        pass


def test_not_ready_raises_on_get():
    m = TrackBufferManager(L=3, max_age_sec=1.0)
    m.update(1, _s(0), 0.0)
    try:
        m.get_window(1)
        assert False
    except KeyError:
        pass
