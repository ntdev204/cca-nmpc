#!/usr/bin/env python3
"""TF transform utilities for converting points between frames."""
from typing import Optional
import rclpy.time
from geometry_msgs.msg import Point, PointStamped, TransformStamped
from rclpy.duration import Duration
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — Registers PointStamped transform


class FrameTransformer:
    """Transform points from camera optical frame to map frame using tf2."""

    def __init__(
        self,
        tf_buffer: tf2_ros.Buffer,
        timeout_sec: float = 0.1
    ) -> None:
        """Initialize transformer.

        Args:
            tf_buffer: tf2_ros.Buffer instance with active TransformListener
            timeout_sec: Max wait time for transform lookup
        """
        self._buffer = tf_buffer
        self._timeout = Duration(seconds=timeout_sec)

    def transform_point_to_map(
        self,
        point: Point,
        source_frame: str,
        target_frame: str,
        stamp: rclpy.time.Time
    ) -> Optional[PointStamped]:
        """Transform point from source frame to target frame.

        Args:
            point: Point in source_frame coordinates
            source_frame: Source frame ID (optical frame)
            target_frame: Target frame ID (map)
            stamp: Timestamp for transform lookup

        Returns:
            PointStamped in target frame, or None if transform unavailable
        """
        point_stamped = PointStamped()
        point_stamped.header.frame_id = source_frame
        point_stamped.header.stamp = stamp.to_msg()
        point_stamped.point = point

        try:
            # Lookup transform from source to target at given timestamp
            transform: TransformStamped = self._buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                self._timeout
            )
            # Apply transform using tf2_geometry_msgs registered method
            transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return transformed
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException
        ) as e:
            # Transform unavailable — caller will skip this detection
            return None
