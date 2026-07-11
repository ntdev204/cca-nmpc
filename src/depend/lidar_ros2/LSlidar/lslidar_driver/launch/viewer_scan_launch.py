#!/usr/bin/python3
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

import os

def generate_launch_description():

    rviz2_config = os.path.join(get_package_share_directory('lslidar_driver'),'rviz','lslidar.rviz')
      
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d',rviz2_config],
        output='screen')                                               

    return LaunchDescription([
        rviz2_node,
    ])
    
