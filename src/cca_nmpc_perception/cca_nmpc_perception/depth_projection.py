#!/usr/bin/env python3
import numpy as np
from typing import Optional
from dataclasses import dataclass
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point


@dataclass(frozen=True)
class Detection2D:
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
    h, w = depth_image.shape[:2]

    cx_norm = (detection.x_min + detection.x_max) / 2.0
    cy_norm = (detection.y_min + detection.y_max) / 2.0
    cx_px = int(cx_norm * w)
    cy_px = int(cy_norm * h)

    cx_px = max(0, min(w - 1, cx_px))
    cy_px = max(0, min(h - 1, cy_px))

    y_min = max(0, cy_px - sampling_radius)
    y_max = min(h, cy_px + sampling_radius + 1)
    x_min = max(0, cx_px - sampling_radius)
    x_max = min(w, cx_px + sampling_radius + 1)

    depth_patch = depth_image[y_min:y_max, x_min:x_max]

    valid_depths = depth_patch[depth_patch > 0]
    if valid_depths.size == 0:
        return None

    median_depth = float(np.median(valid_depths))

    if depth_image.dtype == np.uint16:
        median_depth /= 1000.0

    K = np.array(camera_info.k).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx_intrinsic, cy_intrinsic = K[0, 2], K[1, 2]

    x_3d = (cx_px - cx_intrinsic) * median_depth / fx
    y_3d = (cy_px - cy_intrinsic) * median_depth / fy
    z_3d = median_depth

    point = Point()
    point.x = x_3d
    point.y = y_3d
    point.z = z_3d
    return point
