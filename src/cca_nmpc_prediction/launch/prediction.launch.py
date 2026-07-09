"""Launch the CCA-NMPC prediction node with its parameter file."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('cca_nmpc_prediction')
    default_config = os.path.join(pkg_share, 'config', 'prediction.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to prediction node config YAML',
        ),
        Node(
            package='cca_nmpc_prediction',
            executable='prediction_node',
            name='prediction_node',
            output='screen',
            parameters=[LaunchConfiguration('config_file')],
        ),
    ])
