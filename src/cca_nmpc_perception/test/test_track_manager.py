#!/usr/bin/env python3
"""Tests for track_manager.py (HP-06)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cca_nmpc_perception'))

from cca_nmpc_perception.track_manager import TrackManager


def make_manager(
    process_noise_std=0.1,
    measurement_noise_std=0.15,
    association_distance_gate=2.0,
    max_track_age_sec=1.0
) -> TrackManager:
    return TrackManager(
        process_noise_std=process_noise_std,
        measurement_noise_std=measurement_noise_std,
        association_distance_gate=association_distance_gate,
        max_track_age_sec=max_track_age_sec
    )


class TestNewTrackCreation:

    def test_single_measurement_creates_track(self):
        manager = make_manager()
        tracks = manager.update([(1.0, 2.0)], current_time=0.0)
        assert len(tracks) == 1
        assert tracks[0].track_id == 0

    def test_track_position_near_measurement(self):
        manager = make_manager()
        tracks = manager.update([(3.0, 4.0)], current_time=0.0)
        pos = tracks[0].position
        assert abs(pos[0] - 3.0) < 0.01
        assert abs(pos[1] - 4.0) < 0.01

    def test_track_initial_velocity_zero(self):
        manager = make_manager()
        tracks = manager.update([(1.0, 1.0)], current_time=0.0)
        vel = tracks[0].velocity
        assert abs(vel[0]) < 1e-9
        assert abs(vel[1]) < 1e-9

    def test_multiple_new_measurements_create_separate_tracks(self):
        manager = make_manager(association_distance_gate=0.5)
        tracks = manager.update([(0.0, 0.0), (10.0, 10.0)], current_time=0.0)
        assert len(tracks) == 2
        ids = {t.track_id for t in tracks}
        assert len(ids) == 2

    def test_track_ids_increment(self):
        manager = make_manager(association_distance_gate=0.5)
        manager.update([(0.0, 0.0)], current_time=0.0)
        manager.update([(10.0, 10.0)], current_time=0.1)
        tracks = manager.update([], current_time=0.2)
        # Two tracks still alive, ids are 0 and 1
        ids = sorted(t.track_id for t in tracks)
        assert ids == [0, 1]


class TestAssociation:

    def test_nearby_measurement_updates_existing_track(self):
        manager = make_manager()
        tracks_t0 = manager.update([(0.0, 0.0)], current_time=0.0)
        track_id = tracks_t0[0].track_id

        tracks_t1 = manager.update([(0.1, 0.1)], current_time=0.1)
        assert len(tracks_t1) == 1
        assert tracks_t1[0].track_id == track_id  # same track, not new

    def test_far_measurement_creates_new_track(self):
        manager = make_manager(association_distance_gate=1.0)
        manager.update([(0.0, 0.0)], current_time=0.0)
        tracks = manager.update([(5.0, 5.0)], current_time=0.1)
        assert len(tracks) == 2

    def test_no_measurements_does_not_create_track(self):
        manager = make_manager()
        tracks = manager.update([], current_time=0.0)
        assert len(tracks) == 0

    def test_no_measurements_existing_track_survives_briefly(self):
        manager = make_manager(max_track_age_sec=1.0)
        manager.update([(1.0, 1.0)], current_time=0.0)
        tracks = manager.update([], current_time=0.5)
        assert len(tracks) == 1

    def test_two_tracks_two_measurements_correct_association(self):
        manager = make_manager(association_distance_gate=2.0)
        manager.update([(0.0, 0.0), (10.0, 0.0)], current_time=0.0)
        tracks = manager.update([(0.1, 0.0), (10.1, 0.0)], current_time=0.1)
        assert len(tracks) == 2


class TestStalePruning:

    def test_stale_track_removed_after_max_age(self):
        manager = make_manager(max_track_age_sec=0.5)
        manager.update([(0.0, 0.0)], current_time=0.0)
        tracks = manager.update([], current_time=1.0)  # 1s > 0.5s max age
        assert len(tracks) == 0

    def test_track_survives_within_max_age(self):
        manager = make_manager(max_track_age_sec=1.0)
        manager.update([(0.0, 0.0)], current_time=0.0)
        tracks = manager.update([], current_time=0.9)
        assert len(tracks) == 1

    def test_track_revived_by_measurement_resets_age(self):
        manager = make_manager(max_track_age_sec=0.5)
        manager.update([(0.0, 0.0)], current_time=0.0)
        manager.update([(0.0, 0.0)], current_time=0.4)  # update resets last_update_time
        tracks = manager.update([], current_time=0.8)
        # 0.8 - 0.4 = 0.4 < 0.5 → still alive
        assert len(tracks) == 1

    def test_only_stale_tracks_removed(self):
        manager = make_manager(max_track_age_sec=0.5, association_distance_gate=0.5)
        manager.update([(0.0, 0.0), (10.0, 0.0)], current_time=0.0)
        # Only update track near (0,0), let (10,0) go stale
        manager.update([(0.05, 0.0)], current_time=0.6)
        tracks = manager.update([(0.1, 0.0)], current_time=0.7)
        positions = [t.position for t in tracks]
        xs = [p[0] for p in positions]
        # Only track near x=0 should remain
        assert all(x < 1.0 for x in xs)
        assert len(tracks) == 1


class TestUpdateReturnValues:

    def test_update_returns_list(self):
        manager = make_manager()
        result = manager.update([(1.0, 1.0)], current_time=0.0)
        assert isinstance(result, list)

    def test_update_returns_copy_not_internal(self):
        manager = make_manager()
        result1 = manager.update([(1.0, 1.0)], current_time=0.0)
        result2 = manager.update([(1.1, 1.1)], current_time=0.1)
        # Modifying result1 should not affect internal state
        assert result1 is not result2
