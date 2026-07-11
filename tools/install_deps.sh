#!/bin/bash
# Install Gazebo, Python suite, and rosdep dependencies for CCA-NMPC simulation

set -e

echo "=== CCA-NMPC Dependency Installation ==="

# Source ROS 2 Humble
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "✓ Sourced ROS 2 Humble"
else
    echo "✗ ROS 2 Humble not found. Install ROS 2 Humble first."
    exit 1
fi

# Update rosdep database
echo ""
echo "=== Updating rosdep database ==="
rosdep update

# Install Gazebo Classic (ros-humble-gazebo-ros-pkgs includes Gazebo 11)
echo ""
echo "=== Installing Gazebo Classic + ROS plugins ==="
sudo apt update
sudo apt install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-gazebo-ros2-control-demos \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-xacro \
    ros-humble-rviz2 \
    ros-humble-slam-toolbox \
    ros-humble-nav2-bringup \
    ros-humble-teleop-twist-keyboard

echo "✓ Gazebo Classic + ROS 2 plugins installed"

# Install Python dependencies for perception
echo ""
echo "=== Installing Python suite for perception ==="
pip3 install --upgrade pip
pip3 install \
    numpy \
    opencv-python \
    opencv-contrib-python \
    ultralytics \
    onnx \
    onnxruntime

echo "✓ Python perception suite installed"

# Install rosdep dependencies for workspace
echo ""
echo "=== Installing rosdep dependencies ==="
cd "$(dirname "$0")/.."
rosdep install --from-paths src --ignore-src -r -y

echo ""
echo "=== All dependencies installed ==="
echo ""
echo "Next steps:"
echo "  1. Build workspace: colcon build"
echo "  2. Source workspace: source install/setup.bash"
echo "  3. Launch Gazebo sim: ros2 launch cca_nmpc_bringup gazebo_sim.launch.py"
