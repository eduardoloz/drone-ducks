"""TypeFly robot platform for PX4 drones, built on quackdrone.Drone.

Ported from Eduardo Lozano's typefly/platforms/px4_wrapper.py (ProjectQuack TypeFly branch
eduardo/px4-gazebo, Nov 2026): same skills, camera geometry and approach logic, but all MAVSDK,
threading and safety code now lives in quackdrone, so the drone behaves identically whether it is
driven by TypeFly, a test script or a perception loop.

TypeFly finds this class through a 3-line shim, typefly/platforms/px4_wrapper.py, that
sim/typefly/patches/0001-quack-px4-platform.patch adds to a pinned upstream TypeFly checkout.

    TypeFly skill  move_forward(1)  ->  _move(1, 0)  ->  Drone.move(forward=1)  ->  PX4 Offboard
    TypeFly skill  rotate_left(90)  ->  _rotate(90)  ->  Drone.turn(-90)        (TypeFly: + = left)

robot_info.json "extra" keys (all optional):
  connection        MAVSDK URL, default udpin://0.0.0.0:14540 (SITL). Real drone: serial:///dev/ttyTHS1:921600
  camera_url        any quackdrone.camera source, default http://127.0.0.1:5601 (Gazebo bridge)
  takeoff_alt       metres above the takeoff point, default 1.5
  max_step_m, min_alt, max_alt, fence_radius_m        quackdrone Limits
  pos_tolerance, settle_speed, yaw_tolerance, move_timeout     quackdrone DroneConfig
  yolo_conf (0.35), yolo_tracking (false)
  camera_hfov_deg / camera_pitch_deg / camera_aspect / camera_z_offset   quack_cam geometry
  goto_stop_dist (2.5 m), goto_stop_width (0.6)       when goto() stops approaching
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Optional

from overrides import overrides
from PIL import Image

from typefly.robot_info import RobotInfo
from typefly.robot_wrapper import RobotObservation, RobotWrapper
from typefly.utils import print_t
from typefly.yolo_client import YoloClient

from ..camera import CameraSource, open_camera
from ..drone import Drone, DroneConfig
from ..safety import Limits

# old px4_wrapper key -> quackdrone field name
_RENAMES = {"takeoff_alt": "takeoff_alt_m", "min_alt": "min_alt_m", "max_alt": "max_alt_m",
            "pos_tolerance": "pos_tolerance_m", "settle_speed": "settle_speed_m_s",
            "yaw_tolerance": "yaw_tolerance_deg", "move_timeout": "move_timeout_s"}


def drone_settings(extra: dict) -> dict:
    """Translate robot_info extra keys (old px4_wrapper names accepted) for Drone.from_dict."""
    out = dict(extra or {})
    for old, new in _RENAMES.items():
        if old in out and new not in out:
            out[new] = out[old]
    return out


class PX4Observation(RobotObservation):
    def __init__(self, robot_info: RobotInfo, camera: CameraSource | str, rate: int = 10,
                 capture_hz: float = 15.0, tracking: bool = False, conf: float = 0.35):
        super().__init__(robot_info, rate)
        # TypeFly's default 0.2 lets plain sim geometry (a grey pillar) pass as "stop sign".
        self.conf = conf
        # Tracking off by default: CPU-only YOLO answers ~1 Hz, slower than the tracker's 1 s
        # track lifetime, so tracks would flicker. Sim frames are clean anyway.
        self.yolo_client = YoloClient(robot_info, enable_tracking=tracking)
        self._camera_spec = camera
        self.camera: Optional[CameraSource] = None
        self._detect_busy = False
        self._capture_period = 1.0 / capture_hz
        self.capture_thread = threading.Thread(target=self._capture_spin, name="px4-camera", daemon=True)

    def _capture_spin(self):
        warned = False
        while self.running:
            t0 = time.time()
            frame = None
            try:
                if self.camera is None:
                    self.camera = open_camera(self._camera_spec)
                frame = self.camera.read(timeout_s=1.0)
            except Exception as e:
                frame = None
                if not warned:
                    print_t(f"[PX4] camera {self._camera_spec!r} not available ({e})")
            if frame is not None:
                self._image = Image.fromarray(frame.rgb, "RGB")
                warned = False
            elif not warned:
                print_t(f"[PX4] no frames from {self._camera_spec!r}. Sim: is the camera bridge up "
                        "(sim/scripts/start_sim.sh)?")
                warned = True
                time.sleep(1.0)
            time.sleep(max(0.0, self._capture_period - (time.time() - t0)))

    @overrides
    def _start(self):
        self.capture_thread.start()

    @overrides
    def _stop(self):
        self.capture_thread.join(timeout=2.0)
        if self.camera is not None:
            self.camera.close()

    @overrides
    async def process_image(self, image: Image.Image):
        # The base loop fires this every 1/rate s; keep at most one request in flight so a slow
        # (CPU) YOLO worker isn't buried under a backlog of stale frames.
        if self._detect_busy:
            return
        self._detect_busy = True
        try:
            await self.yolo_client.detect(image, conf=self.conf)
        finally:
            self._detect_busy = False

    @overrides
    def fetch_processed_result(self) -> dict[str, Any]:
        _, object_list = self.yolo_client.latest_result
        return {"yolo": object_list}


class PX4Wrapper(RobotWrapper):
    camera_moves_with_body = True  # forward camera fixed to the airframe

    def __init__(self, robot_info: RobotInfo, drone: Optional[Drone] = None,
                 obs: Optional[RobotObservation] = None):
        extra = drone_settings(robot_info.extra or {})
        self.hfov_deg = float(extra.get("camera_hfov_deg", 68.75))      # quack_cam: 1.2 rad
        self.cam_pitch_deg = float(extra.get("camera_pitch_deg", 8.0))  # down-tilt of quack_cam
        self.cam_aspect = float(extra.get("camera_aspect", 4 / 3))
        self.cam_z_offset = float(extra.get("camera_z_offset", 0.24))   # camera above body origin
        self.goto_stop_dist = float(extra.get("goto_stop_dist", 2.5))
        self.goto_stop_width = float(extra.get("goto_stop_width", 0.6))
        if obs is None:
            obs = PX4Observation(robot_info, extra.get("camera_url", "http://127.0.0.1:5601"),
                                 tracking=bool(extra.get("yolo_tracking", False)),
                                 conf=float(extra.get("yolo_conf", 0.35)))
        super().__init__(robot_info, obs)
        # Constructing never touches the network; start() connects.
        self.drone = drone or Drone.from_dict(extra, log=print_t)
        self.skillset.add_skill(self.lift, "Move up (+) or down (-) by a dist (m)")
        self.skillset.add_skill(self.land, "Land the drone where it is")

    # -------------------------------------------------- lifecycle
    @overrides
    def start(self) -> bool:
        print_t(f"[PX4] connecting on {self.drone.connection} ...")
        self.drone.connect(timeout_s=40)
        self.obs.start()
        t = self.drone.telemetry
        if t.in_air and t.armed:
            print_t("[PX4] drone already flying -> taking over in Offboard")
            self.drone.resume_control()
        else:
            print_t("[PX4] waiting for PX4's estimator, then taking off")
            self.drone.takeoff()
        return True

    @overrides
    def stop(self) -> bool:
        try:
            self.drone.land()
        finally:
            try:
                self.obs.stop()
            except Exception:
                pass
            self.drone.close(land=False)
        return True

    def land(self):
        self._policy_guard("action")  # count + honor cancellation
        self.drone.land()

    # -------------------------------------------------- motion primitives
    def _run(self, start_move) -> None:
        """Send a move (wait=False) then wait for it while honouring TypeFly cancellation."""
        start_move()
        self.drone.wait_reached(poll=self._policy_poll)

    @overrides
    def _move(self, dx: float, dy: float):
        """dx = forward (m), dy = LEFT (m) in TypeFly's convention."""
        print_t(f"[PX4] move fwd={dx:.2f} left={dy:.2f}")
        self._run(lambda: self.drone.move(forward=dx, right=-dy, wait=False))

    @overrides
    def _rotate(self, deg: float):
        """TypeFly: positive deg = turn left (counter-clockwise from above)."""
        print_t(f"[PX4] rotate {deg:+.0f} deg (left positive)")
        self._run(lambda: self.drone.turn(-float(deg), wait=False))

    def lift(self, dist: float):
        dist = self._policy_guard("move", dist)
        print_t(f"[PX4] lift {dist:+.2f} m")
        self._run(lambda: self.drone.move(up=float(dist), wait=False))

    def altitude(self) -> float:
        """Metres above the takeoff point."""
        return self.drone.altitude_m

    # -------------------------------------------------- vision skills tuned for a drone
    @overrides
    def orienting(self, object_name: str) -> bool:
        print_t(f"-> Orient to {object_name}")
        x = self.object_x(object_name)
        if not isinstance(x, (int, float)):
            return False
        offset_deg = (0.5 - x) * self.hfov_deg  # +ve = object is left of centre
        if abs(offset_deg) > 3:
            self.rotate_left(round(offset_deg))
        return True

    def ground_distance(self, info) -> Optional[float]:
        """Horizontal distance (m) to an object standing on the ground, from where its base appears
        in the image: flat-ground pinhole geometry with known camera height and pitch. None if the
        base is above the horizon or cut off by the frame."""
        bottom = info.y + info.h / 2
        if bottom >= 0.98:
            return None
        tan_half_v = math.tan(math.radians(self.hfov_deg) / 2) / self.cam_aspect
        below_axis = math.degrees(math.atan((bottom - 0.5) * 2 * tan_half_v))
        depression = self.cam_pitch_deg + below_axis
        if depression <= 0.5:
            return None
        cam_h = self.altitude() + self.cam_z_offset
        return cam_h / math.tan(math.radians(depression))

    @overrides
    def object_dist(self, object_name: str) -> float:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'dist: {object_name} is not in sight'
        d = self.ground_distance(info)
        return round(d, 2) if d is not None else f'dist: {object_name} is too close to estimate'

    @overrides
    def goto(self, object_name: str) -> bool:
        """Fly toward a visible object: re-centre, step forward (bigger steps while it looks far),
        stop when it is close (wide in the image / base near the bottom edge) or lost."""
        print_t(f"-> Go to {object_name}")
        if not self.orienting(object_name):
            return False
        for _ in range(12):
            info = self.get_obj_info(object_name)
            if info is None:
                print_t(f"[PX4] goto: lost sight of {object_name}; assuming close")
                break
            dist = self.ground_distance(info)
            print_t(f"[PX4] goto: {object_name} x={info.x:.2f} w={info.w:.2f} bottom={info.y + info.h / 2:.2f} "
                    f"est_dist={dist if dist is None else round(dist, 1)}")
            if dist is None or dist <= self.goto_stop_dist or info.w >= self.goto_stop_width:
                break
            self.move_forward(round(max(0.3, min(3.0, dist - self.goto_stop_dist)), 2))
            time.sleep(0.4)  # let a fresh detection arrive after the move
            self.orienting(object_name)
        print_t(f"[PX4] goto: arrived near {object_name}")
        return True

    # -------------------------------------------------- safety
    @overrides
    def stop_motion(self) -> None:
        """Hover where we are (keeps Offboard and altitude). Called when a plan errors/cancels."""
        try:
            if self.drone.in_control:
                self.drone.hold()
        except Exception as ex:
            print_t(f"[PX4] stop_motion failed: {ex}")

    @overrides
    def emergency_shutdown(self) -> None:
        try:
            self.drone.land()
        except Exception as ex:
            print_t(f"[PX4] emergency_shutdown failed: {ex}")
