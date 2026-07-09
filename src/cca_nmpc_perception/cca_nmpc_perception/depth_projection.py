#!/usr/bin/env python3
"""Depth projection from 2D bounding box to 3D point in optical frame.

Samples median valid depth around box center, applies pinhole projection
using CameraInfo intrinsics to produce point in optical frame (NOT camera_link).
"""
import numpy as np
from typing import Optional
from dataclasses import dataclass
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point


@dataclass(frozen=True)
class Detection2D:
    """Normalized 2D bounding box [0,1]."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float


def project_detection_to_3d(
    detection: Detection2D,
    depth_image: np.ndarray,
    camera_info: CameraInfo,
    sampling_radius: int = 5
) -> Optional[Point]:
    """Project 2D detection to 3D point in optical frame.

    Args:
        detection: Normalized [0,1] bounding box
        depth_image: Aligned depth image (H, W), uint16 or float32, depth in mm or m
        camera_info: Camera intrinsics K, distortion model (assume plumb_bob or none)
        sampling_radius: Pixel radius around box center for median depth

    Returns:
        Point in optical frame (camera_info.header.frame_id), or None if no valid depth
    """
    h, w = depth_image.shape[:2]

    # Denormalize box center to pixel coords
    cx_norm = (detection.x_min + detection.x_max) / 2.0
    cy_norm = (detection.y_min + detection.y_max) / 2.0
    cx_px = int(cx_norm * w)
    cy_px = int(cy_norm * h)

    # Clamp to image bounds
    cx_px = max(0, min(w - 1, cx_px))
    cy_px = max(0, min(h - 1, cy_px))

    # Sample depth in radius around center
    y_min = max(0, cy_px - sampling_radius)
    y_max = min(h, cy_px + sampling_radius + 1)
    x_min = max(0, cx_px - sampling_radius)
    x_max = min(w, cx_px + sampling_radius + 1)

    depth_patch = depth_image[y_min:y_max, x_min:x_max]

    # Filter valid depth (>0)
    valid_depths = depth_patch[depth_patch > 0]
    if valid_depths.size == 0:
        return None

    # Median depth (robust to outliers)
    median_depth = float(np.median(valid_depths))

    # Convert mm to m if needed (assume uint16 = mm, float32 = m)
    if depth_image.dtype == np.uint16:
        median_depth /= 1000.0

    # Pinhole projection (optical frame: Z forward, X right, Y down)
    # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    K = np.array(camera_info.k).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx_intrinsic, cy_intrinsic = K[0, 2], K[1, 2]

    # Back-project pixel to 3D ray in optical frame
    x_3d = (cx_px - cx_intrinsic) * median_depth / fx
    y_3d = (cy_px - cy_intrinsic) * median_depth / fy
    z_3d = median_depth

    point = Point()
    point.x = x_3d
    point.y = y_3d
    point.z = z_3d
    return point
