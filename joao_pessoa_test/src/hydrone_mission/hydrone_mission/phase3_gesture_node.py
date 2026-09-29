#!/usr/bin/env python3
"""
phase3_gesture_node — thin mission node for Phase 3 (Human-Swarm Interaction).

Reuses the existing stack instead of talking to MAVROS directly:
  - hydrone_controller for arm/takeoff/land (services) and velocity setpoints
    (/hydrone/controller/cmd_vel, geometry_msgs/Twist, frame_id "base_link":
    +x forward, +y left, +z up, angular.z = yaw rate — see hydrone_controller
    docstring).
  - hydrone_vision's HumanGesture (/hydrone/vision/human_gesture) for the
    gesture classification (gesture_core.classify, ported from frl_core.py).

Sequence (matches the rules: operator stands mid-arena, drone approaches
autonomously, then everything else is gesture-driven):

  IDLE -> ARM -> TAKEOFF -> CREEP (blind, calibrated, low speed, forward)
       -> TURN (blind, calibrated, 90 deg) -> [SEARCH if nobody seen yet]
       -> GESTURE (indefinite, driven by HumanGesture) -> POUSAR (sustained)
       -> LAND -> DONE

CRITICAL OPEN RISK (see chat): without GPS/LIO for Phase 3, the EKF has no
external position/velocity source. hydrone_controller's cmd_vel path sends
SET_POSITION_TARGET_LOCAL_NED (GUIDED). Whether that arms and behaves sanely
without an EKF position source is UNVERIFIED — test in SITL with
mode=GUIDED_NOGPS before flying. `mode_for_flight` param lets you pick the
mode without touching this file.

Any blind (CREEP/TURN) leg is a rough calibration, not a guarantee — tune
`creep_distance`/`creep_speed`/`turn_deg`/`turn_rate` against real flights in
the arena, on different battery levels.

Place at: src/hydrone_mission/hydrone_mission/phase3_gesture_node.py
Register in src/hydrone_mission/setup.py entry_points, e.g.:
    "phase3_gesture_node = hydrone_mission.phase3_gesture_node:main"
"""

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from hydrone_msgs.msg import HumanGesture, MissionState

from hydrone_vision.gesture_core import Debouncer


# body-frame (base_link) unit vector per confirmed gesture: (fwd, left, up, yaw_rate)
GESTURE_VEC = {
    "HOVER":     (0, 0, 0, 0),
    "STOP":      (0, 0, 0, 0),
    "POUSAR":    (0, 0, 0, 0),
    "SUBIR":     (0, 0, 1, 0),
    "DESCER":    (0, 0, -1, 0),
    "APROXIMAR": (1, 0, 0, 0),
    "AFASTAR":   (-1, 0, 0, 0),
    "ESQUERDA":  (0, 1, 0, 0),
    "DIREITA":   (0, -1, 0, 0),
}


class Phase3GestureNode(Node):

    def __init__(self):
        super().__init__('phase3_gesture_node')
        p = self.declare_parameters('', [
            ('mode_for_flight', 'GUIDED_NOGPS'),   # 'GUIDED' or 'GUIDED_NOGPS' — TEST IN SITL FIRST
            ('takeoff_alt', 1.2),
            ('creep_distance', 2.0),
            ('creep_speed', 0.2),
            ('turn_deg', 90.0),
            ('turn_sign', -1.0),        # -1 = direita (clockwise, visto de cima)
            ('turn_rate_deg_s', 15.0),
            ('gesture_speed', 0.25),    # m/s por comando confirmado
            ('gesture_yaw_rate', 0.0),  # rad/s — 0: a pessoa se reposiciona, o drone não gira sozinho
            ('gesture_timeout', 1.0),   # s sem HumanGesture -> força HOVER
            ('search_enabled', True),
            ('search_yaw_rate_deg_s', 10.0),
            ('search_timeout', 15.0),
            ('abort_creep_on_detection', True),
            ('min_gesture_confidence', 0.5),
            ('cmd_hz', 10.0),
            ('pos_tolerance', 0.15),
            ('service_timeout', 5.0),
        ])
        self.par = {q.name: q.value for q in p}

        self.state = 'IDLE'
        self.mav_state = State()
        self.pose = None            # PoseStamped from hydrone_controller/drone_pose
        self.armed_z = None
        self.t_state0 = None        # wall clock at state entry
        self.gesture = None         # last HumanGesture msg
        self.t_last_gesture = None
        self.debounce = Debouncer()
        self.t_search_dir = 1.0

        self.create_subscription(State, '/mavros/state', self._cb_state, 10)
        self.create_subscription(PoseStamped, '/hydrone/controller/drone_pose',
                                 self._cb_pose, qos_profile_sensor_data)
        self.create_subscription(HumanGesture, '/hydrone/vision/human_gesture',
                                 self._cb_gesture, 10)

        self.pub_cmd_vel = self.create_publisher(Twist, '/hydrone/controller/cmd_vel', 10)
        self.pub_mission_state = self.create_publisher(MissionState, '/hydrone/mission_state', 10)
        self.pub_debug = self.create_publisher(String, '/hydrone/phase3/state', 10)

        self.cli_arm = self.create_client(Trigger, '/hydrone/controller/arm')
        self.cli_takeoff = self.create_client(SetBool, '/hydrone/controller/takeoff')
        self.cli_land = self.create_client(Trigger, '/hydrone/controller/land')
        self.cli_mode = self.create_client(SetMode, '/mavros/set_mode')

        self.create_timer(1.0, self._announce_phase)
        self._announce_phase()
        self.create_timer(1.0 / max(self.par['cmd_hz'], 1.0), self._tick)
        self.get_logger().info('phase3_gesture_node up — waiting for controller pose')

    # ── inputs ──────────────────────────────────────────────────────────────
    def _cb_state(self, msg):
        self.mav_state = msg

    def _cb_pose(self, msg):
        self.pose = msg

    def _cb_gesture(self, msg: HumanGesture):
        if msg.confidence < self.par['min_gesture_confidence']:
            return
        self.gesture = msg
        self.t_last_gesture = self._now()

    def _announce_phase(self):
        """Tell hydrone_vision to run the gesture pipeline (it gates on
        MissionState.phase == 3). Field name ASSUMED as `phase`; check
        hydrone_msgs/msg/MissionState.msg if this doesn't build."""
        m = MissionState()
        try:
            m.phase = 3
        except AttributeError:
            self.get_logger().error(
                "MissionState has no 'phase' field — check hydrone_msgs/msg/MissionState.msg "
                "and fix this line by hand.")
            return
        self.pub_mission_state.publish(m)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _enter(self, state):
        if state != self.state:
            self.get_logger().info(f'[phase3] {self.state} -> {state}')
            self.state, self.t_state0 = state, self._now()
        self.pub_debug.publish(String(data=self.state))

    def _elapsed(self):
        return self._now() - (self.t_state0 or self._now())

    def _publish_twist(self, fwd, left, up, yaw_rate):
        t = Twist()
        t.linear.x, t.linear.y, t.linear.z = float(fwd), float(left), float(up)
        t.angular.z = float(yaw_rate)
        self.pub_cmd_vel.publish(t)

    def _call(self, cli, req, what, blocking=True):
        if not cli.wait_for_service(timeout_sec=self.par['service_timeout']):
            self.get_logger().error(f'{what}: service unavailable')
            return None
        fut = cli.call_async(req)
        if blocking:
            rclpy.spin_until_future_complete(self, fut, timeout_sec=self.par['service_timeout'])
            return fut.result()
        return fut

    def _human_visible(self):
        if self.t_last_gesture is None:
            return False
        return (self._now() - self.t_last_gesture) < self.par['gesture_timeout']

    # ── state machine ───────────────────────────────────────────────────────
    def _tick(self):
        if self.state == 'IDLE':
            self._enter('SET_MODE')

        elif self.state == 'SET_MODE':
            if self.mav_state.mode != self.par['mode_for_flight']:
                self._call(self.cli_mode, SetMode.Request(
                    custom_mode=self.par['mode_for_flight']), 'set_mode', blocking=False)
                return
            self._enter('ARM')

        elif self.state == 'ARM':
            if not self.mav_state.armed:
                self._call(self.cli_arm, Trigger.Request(), 'arm', blocking=False)
                return
            if self.pose is not None:
                self.armed_z = self.pose.pose.position.z
            self._enter('TAKEOFF')

        elif self.state == 'TAKEOFF':
            if self._elapsed() < 0.1:   # fire once per entry
                self._call(self.cli_takeoff, SetBool.Request(data=False), 'takeoff', blocking=False)
            if self.pose is None or self.armed_z is None:
                return
            target = self.armed_z + self.par['takeoff_alt']
            if abs(self.pose.pose.position.z - target) < self.par['pos_tolerance']:
                self._enter('CREEP')
            elif self._elapsed() > 15.0:
                self.get_logger().warn('takeoff timeout, creeping anyway')
                self._enter('CREEP')

        elif self.state == 'CREEP':
            if self.par['abort_creep_on_detection'] and self._human_visible():
                self._publish_twist(0, 0, 0, 0)
                self._enter('GESTURE')
                return
            duration = self.par['creep_distance'] / max(self.par['creep_speed'], 1e-3)
            if self._elapsed() >= duration:
                self._publish_twist(0, 0, 0, 0)
                self._enter('TURN')
                return
            self._publish_twist(self.par['creep_speed'], 0, 0, 0)

        elif self.state == 'TURN':
            rate = math.radians(self.par['turn_rate_deg_s']) * self.par['turn_sign']
            duration = math.radians(self.par['turn_deg']) / max(abs(rate), 1e-3)
            if self._elapsed() >= duration:
                self._publish_twist(0, 0, 0, 0)
                self._enter('SEARCH' if self.par['search_enabled'] and not self._human_visible()
                            else 'GESTURE')
                return
            self._publish_twist(0, 0, 0, rate)

        elif self.state == 'SEARCH':
            if self._human_visible():
                self._publish_twist(0, 0, 0, 0)
                self._enter('GESTURE')
                return
            if self._elapsed() >= self.par['search_timeout']:
                self.get_logger().warn('operator not found — holding in place, check framing')
                self._publish_twist(0, 0, 0, 0)
                self._enter('GESTURE')
                return
            # sweep back and forth slowly instead of spinning one way forever
            if int(self._elapsed()) % 6 == 0:
                self.t_search_dir *= -1
            rate = math.radians(self.par['search_yaw_rate_deg_s']) * self.t_search_dir
            self._publish_twist(0, 0, 0, rate)

        elif self.state == 'GESTURE':
            self._gesture_step()

        elif self.state == 'LANDING':
            if self._elapsed() < 0.1:
                self._call(self.cli_land, Trigger.Request(), 'land', blocking=False)
            if not self.mav_state.armed:
                self._enter('DONE')
            elif self._elapsed() > 20.0:
                self.get_logger().error('land did not disarm in time')
                self._enter('DONE')

        elif self.state == 'DONE':
            pass

    def _gesture_step(self):
        if not self._human_visible():
            self._publish_twist(0, 0, 0, 0)
            return
        name = self.debounce.update(self.gesture.gesture_name, self._now())
        cmd = name.command
        self.get_logger().info(f'[phase3] comando ativo: {cmd}', throttle_duration_sec=1.0)
        if cmd == 'POUSAR' and name.changed:
            self._publish_twist(0, 0, 0, 0)
            self._enter('LANDING')
            return
        fwd, left, up, _ = GESTURE_VEC.get(cmd, (0, 0, 0, 0))
        s = self.par['gesture_speed']
        self._publish_twist(fwd * s, left * s, up * s, self.par['gesture_yaw_rate'])


def main():
    rclpy.init()
    node = Phase3GestureNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()