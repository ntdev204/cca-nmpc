#!/usr/bin/env python3
"""PerceptionNode: synchronized RGB/depth/camera_info callback pipeline.

Detects humans with YOLO, projects to 3D optical frame, transforms to map,
tracks with Kalman filter, publishes HumanStateArray.
"""
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from pathlib import Path
import cv_bridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros

from cca_nmpc_msgs.msg import HumanState, HumanStateArray
from .detector import TensorRtYoloDetector
from .depth_projection import Detection2D, project_detection_to_3d
from .tf_transform import FrameTransformer
from .track_manager import TrackManager


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self._declare_and_load_params()
        self._validate_params()

        # Build detector
        self._detector = self._build_detector()

        # TF
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._frame_transformer = FrameTransformer(self._tf_buffer)

        # Tracker
        self._tracker = TrackManager(
            process_noise_std=self.process_noise_std,
            measurement_noise_std=self.measurement_noise_std,
            association_distance_gate=self.association_distance_gate,
            max_track_age_sec=self.max_track_age_sec
        )

        # Publisher
        self._pub = self.create_publisher(HumanStateArray, '/human_states', 10)

        # cv_bridge
        self._bridge = cv_bridge.CvBridge()

        # Synchronized subscribers
        self._rgb_sub = message_filters.Subscriber(self, Image, self.rgb_image_topic, qos_profile=qos_profile_sensor_data)
        self._depth_sub = message_filters.Subscriber(self, Image, self.depth_image_topic, qos_profile=qos_profile_sensor_data)
        self._info_sub = message_filters.Subscriber(self, CameraInfo, self.camera_info_topic, qos_profile=qos_profile_sensor_data)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub, self._info_sub],
            queue_size=10,
            slop=0.05
        )
        self._sync.registerCallback(self._sync_callback)

        self.get_logger().info('PerceptionNode started')
        self.get_logger().info(f'  RGB: {self.rgb_image_topic}')
        self.get_logger().info(f'  Depth: {self.depth_image_topic}')
        self.get_logger().info('  Output: /human_states')

    def _declare_and_load_params(self) -> None:
        self.declare_parameter('yolo_engine_path', 'models/yolo26m_human.engine')
        self.declare_parameter('detection_confidence_threshold', 0.4)
        self.declare_parameter('max_track_age_sec', 1.0)
        self.declare_parameter('association_distance_gate', 2.0)
        self.declare_parameter('kalman.process_noise_std', 0.1)
        self.declare_parameter('kalman.measurement_noise_std', 0.15)
        self.declare_parameter('rgb_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('sensor_qos', 'sensor_data')
        self.declare_parameter('depth_sampling_radius', 5)
        self.declare_parameter('camera_optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('require_depth_alignment', True)

        self.yolo_engine_path = self.get_parameter('yolo_engine_path').value
        self.detection_confidence_threshold = self.get_parameter('detection_confidence_threshold').value
        self.max_track_age_sec = self.get_parameter('max_track_age_sec').value
        self.association_distance_gate = self.get_parameter('association_distance_gate').value
        self.process_noise_std = self.get_parameter('kalman.process_noise_std').value
        self.measurement_noise_std = self.get_parameter('kalman.measurement_noise_std').value
        self.rgb_image_topic = self.get_parameter('rgb_image_topic').value
        self.depth_image_topic = self.get_parameter('depth_image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.sensor_qos = self.get_parameter('sensor_qos').value
        self.depth_sampling_radius = self.get_parameter('depth_sampling_radius').value
        self.camera_optical_frame = self.get_parameter('camera_optical_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.require_depth_alignment = self.get_parameter('require_depth_alignment').value

    def _validate_params(self) -> None:
        engine_path = Path(self.yolo_engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f'Missing YOLO engine: {self.yolo_engine_path}')
        if not engine_path.is_file():
            raise ValueError(f'YOLO engine must be a file: {self.yolo_engine_path}')

        if self.require_depth_alignment:
            depth_lower = self.depth_image_topic.lower()
            is_aligned = any(k in depth_lower for k in ('aligned', 'depth_to_color', 'registered'))
            if not is_aligned:
                self.get_logger().error(
                    f'require_depth_alignment=true but depth topic not aligned: {self.depth_image_topic}'
                )
                raise ValueError(f'Depth topic must indicate alignment: {self.depth_image_topic}')

    def _build_detector(self):
        """Build the explicitly selected detector; production never falls back."""
        detector = TensorRtYoloDetector(
            self.yolo_engine_path,
            self.detection_confidence_threshold
        )
        self.get_logger().info('Using TensorRtYoloDetector')
        return detector

    def _sync_callback(
        self,
        rgb_msg: Image,
        depth_msg: Image,
        info_msg: CameraInfo
    ) -> None:
        """Synchronized callback: detect → project → transform → track → publish."""
        t_start = time.monotonic()

        # Convert ROS images to numpy
        try:
            rgb_image = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except cv_bridge.CvBridgeError as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        t_decode = time.monotonic()

        # Detect humans
        raw_detections = self._detector.detect(rgb_image)
        t_detect = time.monotonic()

        # Project to 3D in optical frame, then transform to map
        stamp = rgb_msg.header.stamp
        ros_time = rclpy.time.Time.from_msg(stamp)
        measurements: list[tuple[float, float, float]] = []

        for det in raw_detections:
            if det.confidence < self.detection_confidence_threshold:
                continue

            det2d = Detection2D(
                x_min=det.x1, y_min=det.y1,
                x_max=det.x2, y_max=det.y2,
                confidence=det.confidence
            )

            point_optical = project_detection_to_3d(
                det2d, depth_image, info_msg, self.depth_sampling_radius
            )
            if point_optical is None:
                continue

            # Transform to map frame
            point_map = self._frame_transformer.transform_point_to_map(
                point_optical,
                source_frame=info_msg.header.frame_id,
                target_frame=self.map_frame,
                stamp=ros_time
            )
            if point_map is None:
                continue

            measurements.append(
                (point_map.point.x, point_map.point.y, det.confidence)
            )

        t_project = time.monotonic()

        # Update tracker
        current_time = float(ros_time.nanoseconds) / 1e9
        active_tracks = self._tracker.update(measurements, current_time)

        t_track = time.monotonic()

        # Build and publish HumanStateArray
        msg = HumanStateArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame

        for track in active_tracks:
            if not track.is_observed:
                continue
            state = HumanState()
            state.header.stamp = stamp
            state.header.frame_id = self.map_frame
            state.track_id = track.track_id
            state.x = track.position[0]
            state.y = track.position[1]
            state.vx = track.velocity[0]
            state.vy = track.velocity[1]
            state.confidence = track.confidence
            msg.humans.append(state)

        self._pub.publish(msg)
        t_publish = time.monotonic()

        self.get_logger().debug(
            f'Pipeline: decode={1000*(t_decode-t_start):.1f}ms '
            f'detect={1000*(t_detect-t_decode):.1f}ms '
            f'project+tf={1000*(t_project-t_detect):.1f}ms '
            f'track={1000*(t_track-t_project):.1f}ms '
            f'publish={1000*(t_publish-t_track):.1f}ms '
            f'total={1000*(t_publish-t_start):.1f}ms '
            f'tracks={len(active_tracks)}'
        )


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PerceptionNode()
        rclpy.spin(node)
    except (FileNotFoundError, ValueError) as e:
        rclpy.logging.get_logger('perception_node').error(str(e))
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
