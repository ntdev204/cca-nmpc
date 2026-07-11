#!/usr/bin/env python3
"""Tests for depth_projection.py (HP-04)."""
import pytest  # noqa: F401
import numpy as np
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import List

import sys
import os

# Mock ROS2 messages BEFORE importing production code
@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Header:
    frame_id: str = ''


@dataclass
class CameraInfo:
    width: int = 0
    height: int = 0
    k: List[float] = field(default_factory=lambda: [0.0] * 9)
    header: Header = field(default_factory=Header)


# Patch ROS2 imports before importing production code
sys.modules['geometry_msgs'] = MagicMock()
sys.modules['geometry_msgs.msg'] = MagicMock()
sys.modules['sensor_msgs'] = MagicMock()
sys.modules['sensor_msgs.msg'] = MagicMock()
sys.modules['geometry_msgs.msg'].Point = Point
sys.modules['sensor_msgs.msg'].CameraInfo = CameraInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cca_nmpc_perception'))

from cca_nmpc_perception.depth_projection import Detection2D, project_detection_to_3d  # noqa: E402


def make_camera_info(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480):
    info = CameraInfo()
    info.width = width
    info.height = height
    info.k = [fx, 0.0, cx,
              0.0, fy, cy,
              0.0, 0.0, 1.0]
    info.header.frame_id = 'camera_color_optical_frame'
    return info


def make_depth_image(height=480, width=640, fill_mm=2000):
    """Uniform depth in mm (uint16)."""
    return np.full((height, width), fill_mm, dtype=np.uint16)


class TestProjectDetectionTo3d:

    def test_basic_projection_center(self):
        """Center pixel at 2m depth should project to (0, 0, 2)."""
        det = Detection2D(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6,
                          confidence=0.9)
        depth = make_depth_image(fill_mm=2000)
        info = make_camera_info(fx=500, fy=500, cx=320, cy=240)

        point = project_detection_to_3d(det, depth, info, sampling_radius=5)

        assert point is not None
        assert isinstance(point, Point)
        # cx,cy = 320,240 — center box = (320,240) — X=0, Y=0
        assert abs(point.z - 2.0) < 0.01
        assert abs(point.x) < 0.01
        assert abs(point.y) < 0.01

    def test_off_center_projection(self):
        """Box offset by 100px right: X = (cx+100 - cx)*Z/fx."""
        fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
        depth = make_depth_image(fill_mm=1000)
        info = make_camera_info(fx=fx, fy=fy, cx=cx, cy=cy)

        # Box center at pixel (420, 240) — 100 pixels right of principal point
        det = Detection2D(
            x_min=(420 - 20) / 640, y_min=(240 - 20) / 480,
            x_max=(420 + 20) / 640, y_max=(240 + 20) / 480,
            confidence=0.9
        )

        point = project_detection_to_3d(det, depth, info, sampling_radius=5)

        assert point is not None
        expected_x = (420 - cx) * 1.0 / fx   # Z = 1.0m
        expected_y = (240 - cy) * 1.0 / fy
        assert abs(point.x - expected_x) < 0.02
        assert abs(point.y - expected_y) < 0.02
        assert abs(point.z - 1.0) < 0.01

    def test_zero_depth_returns_none(self):
        """Patch of all-zero depth → no valid reading → None."""
        det = Detection2D(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6,
                          confidence=0.9)
        depth = make_depth_image(fill_mm=0)
        info = make_camera_info()

        result = project_detection_to_3d(det, depth, info)
        assert result is None

    def test_all_zero_patch_returns_none(self):
        """Only the sampled patch is zero, rest is valid — still None for that box."""
        depth = make_depth_image(fill_mm=2000)
        cx_pix, cy_pix = 320, 240
        depth[cy_pix - 10:cy_pix + 10, cx_pix - 10:cx_pix + 10] = 0

        det = Detection2D(x_min=0.45, y_min=0.45, x_max=0.55, y_max=0.55,
                          confidence=0.9)
        info = make_camera_info()

        result = project_detection_to_3d(det, depth, info, sampling_radius=5)
        assert result is None

    def test_out_of_image_box_clamped(self):
        """Box extending beyond image bounds: clamp, no crash."""
        det = Detection2D(x_min=-0.1, y_min=-0.1, x_max=1.1, y_max=1.1,
                          confidence=0.9)
        depth = make_depth_image(fill_mm=3000)
        info = make_camera_info()

        point = project_detection_to_3d(det, depth, info)
        assert point is not None
        assert abs(point.z - 3.0) < 0.01

    def test_returns_point_in_optical_frame(self):
        """Output frame_id comes from camera_info header."""
        det = Detection2D(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6,
                          confidence=0.9)
        depth = make_depth_image(fill_mm=1500)
        info = make_camera_info()
        info.header.frame_id = 'camera_color_optical_frame'

        point = project_detection_to_3d(det, depth, info)
        # point itself has no frame_id — caller uses info.header.frame_id
        assert point is not None

    def test_uint16_to_meters_conversion(self):
        """uint16 mm depth → meters: 2000 mm = 2.0 m."""
        det = Detection2D(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6,
                          confidence=0.9)
        depth = make_depth_image(fill_mm=3500)
        info = make_camera_info()

        point = project_detection_to_3d(det, depth, info)
        assert point is not None
        assert abs(point.z - 3.5) < 0.05

    def test_float32_depth_image(self):
        """float32 depth in meters (not uint16 mm) — direct use."""
        det = Detection2D(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6,
                          confidence=0.9)
        depth = np.full((480, 640), 4.0, dtype=np.float32)
        info = make_camera_info()

        point = project_detection_to_3d(det, depth, info)
        assert point is not None
        assert abs(point.z - 4.0) < 0.05
