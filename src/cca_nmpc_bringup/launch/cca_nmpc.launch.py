from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    default_params = PathJoinSubstitution(
        [FindPackageShare("cca_nmpc_bringup"), "config", "cca_nmpc_params.yaml"]
    )
    params = LaunchConfiguration("params_file")
    nodes = [
        ("cca_nmpc_perception", "perception_node", "perception_node"),
        ("cca_nmpc_prediction", "prediction_node", "prediction_node"),
        ("cca_nmpc_context", "context_node", "context_node"),
        ("cca_nmpc_adaptive_params", "adaptive_param_node", "adaptive_param_node"),
        ("cca_nmpc_control", "nmpc_controller_node", "nmpc_controller_node"),
    ]
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        *[
            Node(package=pkg, executable=exe, name=name, output="screen", parameters=[params])
            for pkg, exe, name in nodes
        ],
    ])
