#!/usr/bin/env python3
from typing import Optional
import rclpy.time
from geometry_msgs.msg import Point, PointStamped, TransformStamped
from rclpy.duration import Duration
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — Registers PointStamped transform


class FrameTransformer:
    def __init__(
        self,
        tf_buffer: tf2_ros.Buffer,
        timeout_sec: float = 0.1
    ) -> None:
        self._buffer = tf_buffer
        self._timeout = Duration(seconds=timeout_sec)

    def transform_point_to_map(
        self,
        point: Point,
        source_frame: str,
        target_frame: str,
        stamp: rclpy.time.Time
    ) -> Optional[PointStamped]:
        point_stamped = PointStamped()
        point_stamped.header.frame_id = source_frame
        point_stamped.header.stamp = stamp.to_msg()
        point_stamped.point = point

        try:
            transform: TransformStamped = self._buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                self._timeout
            )
            transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return transformed
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException
        ):
            return None
