#!/usr/bin/env python3
"""Launch perception_node with parameter overrides.

CRITICAL: Astra camera must publish aligned depth OR be launched with
depth_registration:=true. Default Astra topics are NOT aligned.

Example:
  ros2 launch astra_camera astra.launch.py depth_registration:=true
  ros2 launch cca_nmpc_perception human_perception.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('cca_nmpc_perception')
    default_config = os.path.join(pkg_share, 'config', 'human_perception.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to perception config YAML'
        ),
        DeclareLaunchArgument(
            'yolo_engine_path',
            default_value='',
            description='Override YOLO .engine path (empty = use config)'
        ),
        DeclareLaunchArgument(
            'rgb_image_topic',
            default_value='',
            description='Override RGB topic (empty = use config)'
        ),
        DeclareLaunchArgument(
            'depth_image_topic',
            default_value='',
            description='Override depth topic (empty = use config)'
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='',
            description='Override camera_info topic (empty = use config)'
        ),
        DeclareLaunchArgument(
            'map_frame',
            default_value='',
            description='Override map frame (empty = use config)'
        ),

        Node(
            package='cca_nmpc_perception',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    'yolo_engine_path': LaunchConfiguration('yolo_engine_path'),
                    'rgb_image_topic': LaunchConfiguration('rgb_image_topic'),
                    'depth_image_topic': LaunchConfiguration('depth_image_topic'),
                    'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                    'map_frame': LaunchConfiguration('map_frame'),
                }
            ],
            remappings=[
                ('/human_states', '/human_states'),
            ]
        )
    ])
