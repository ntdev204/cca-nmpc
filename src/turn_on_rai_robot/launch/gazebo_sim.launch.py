#!/usr/bin/env python3

import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
import launch_ros.actions
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    bringup_dir = get_package_share_directory("turn_on_rai_robot")
    launch_dir = os.path.join(bringup_dir, "launch")
    world_file = Path(bringup_dir, "worlds", "empty_indoor.world")
    urdf_file = Path(bringup_dir, "urdf", "mini_mec_gazebo.urdf.xacro")
    robot_description = ParameterValue(
        Command(["xacro ", str(urdf_file)]), value_type=str
    )

    imu_config = Path(bringup_dir, "config", "imu.yaml")

    gazebo_server = ExecuteProcess(
        cmd=["gzserver", "-s", "libgazebo_ros_init.so", "-s", "libgazebo_ros_factory.so", str(world_file)],
        output="screen",
    )

    gazebo_client = ExecuteProcess(
        cmd=["gzclient"],
        output="screen",
    )

    spawn_robot = launch_ros.actions.Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "mini_mec_robot",
            "-topic", "robot_description",
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.15",
        ],
        output="screen",
    )

    robot_state_publisher = launch_ros.actions.Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
    )

    base_to_link = launch_ros.actions.Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_link",
        arguments=["0", "0", "0", "0", "0", "0", "base_footprint", "base_link"],
    )

    base_to_gyro = launch_ros.actions.Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_gyro",
        arguments=["0", "0", "0", "0", "0", "0", "base_footprint", "gyro_link"],
    )

    joint_state_publisher = launch_ros.actions.Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
    )

    imu_filter = launch_ros.actions.Node(
        package="imu_filter_madgwick",
        executable="imu_filter_madgwick_node",
        parameters=[str(imu_config)],
    )

    robot_ekf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, "rai_ekf.launch.py")),
        launch_arguments={"carto_slam": "false"}.items(),
    )

    smoke_test_warning = LogInfo(
        msg=("[gazebo_sim] SMOKE-TEST ONLY: diff-drive plugin ignores "
             "cmd_vel.linear.y. Not valid for Mecanum lateral-motion validation.")
    )

    return LaunchDescription([
        smoke_test_warning,
        gazebo_server,
        gazebo_client,
        spawn_robot,
        robot_state_publisher,
        base_to_link,
        base_to_gyro,
        joint_state_publisher,
        imu_filter,
        robot_ekf,
    ])
