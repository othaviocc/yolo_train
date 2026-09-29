"""Phase 3 gesture mission using the Phase 4 KopisX8 simulation stack.

The Phase 4 launch supplies SITL, MAVROS and LIO; this launch adds the
controller, the simulated RGB camera consumer and the gesture mission.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_pkg = get_package_share_directory('hydrone_bringup')
    phase4_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'phase4_sim.launch.py')),
        launch_arguments={
            'agent_name': 'KopisX8',
            'phase': '3',
            'ext_nav': 'lio',
            'mission': 'none',
            'measure_drift': 'false',
        }.items(),
    )

    controller = Node(
        package='hydrone_controller', executable='controller_node',
        output='screen',
        parameters=[{
            'takeoff_height': LaunchConfiguration('takeoff_alt'),
        }],
    )

    vision = Node(
        package='hydrone_vision', executable='vision_node',
        output='screen',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'depth_topic': '',          # gestures do not require depth
            'initial_phase': 3,
            'debug_image': True,
        }],
    )

    phase3 = Node(
        package='hydrone_mission', executable='phase3_gesture_node',
        output='screen',
        parameters=[{
            'takeoff_alt': LaunchConfiguration('takeoff_alt'),
            'creep_distance': LaunchConfiguration('creep_distance'),
            'creep_speed': LaunchConfiguration('creep_speed'),
            'turn_deg': LaunchConfiguration('turn_deg'),
            'mode_for_flight': LaunchConfiguration('mode_for_flight'),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('camera_topic', default_value='/biguasim/uav0_id0/RGBCamera',
                              description='KopisX8 simulated RGB camera image topic.'),
        DeclareLaunchArgument('takeoff_alt', default_value='1.2'),
        DeclareLaunchArgument('creep_distance', default_value='2.0'),
        DeclareLaunchArgument('creep_speed', default_value='0.2'),
        DeclareLaunchArgument('turn_deg', default_value='90.0'),
        DeclareLaunchArgument('mode_for_flight', default_value='GUIDED',
                              description='GUIDED uses the Phase 4 LIO external navigation.'),
        phase4_launch, controller, vision, phase3,
    ])