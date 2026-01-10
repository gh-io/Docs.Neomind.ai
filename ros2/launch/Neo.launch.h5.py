from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='neurobot_ros2',
            executable='lidar_node',
            name='lidar_node'
        ),
        Node(
            package='neurobot_ros2',
            executable='imu_node',
            name='imu_node'
        ),
        Node(
            package='neurobot_ros2',
            executable='camera_node',
            name='camera_node'
        ),
        Node(
            package='neurobot_ros2',
            executable='motor_node',
            name='motor_node'
        ),
        Node(
            package='neurobot_ros2',
            executable='swarm_node',
            name='swarm_node'
        ),
    ])
