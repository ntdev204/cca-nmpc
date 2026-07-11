import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('cca_nmpc_context')
    default_config = os.path.join(pkg_share, 'config', 'context.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to context node config YAML',
        ),
        Node(
            package='cca_nmpc_context',
            executable='context_node',
            name='context_node',
            output='screen',
            parameters=[LaunchConfiguration('config_file')],
        ),
    ])
