"""Lab 9 — the two-object loco-manipulation scene with egocentric cameras.

Lab 8's capstone scene with three changes, each of which exists to make a
specific measurement possible.

**Two objects, not one.** A red cup and a blue box stand side by side on the
pick pedestal, and which one is nearer the robot is randomised per seed. The
instruction names the target. With one object, a policy conditioned on four
task labels can infer the task from the robot's own pose — walking, reaching,
carrying and placing look nothing alike — and ignore the language entirely.
Two objects make the same image demand different actions under different
instructions, which is the only setup in which "the policy follows
instructions" is a falsifiable claim (tasks/LESSONS.md § L-P0-c).

**Egocentric cameras.** A head camera on `torso_link` and a wrist camera on
`right_wrist_yaw_link`. The head view is what tells the policy where the
objects are; the wrist view is what makes the last few centimetres of a reach
observable at all, because at 128 px the objects are a handful of pixels in the
head view by the time the hand is near them.

**Per-seed randomisation.** Object placement, which object is nearer, object
hue within its colour family, and light position. The ranges are deliberately
narrow enough that Lab 8's controller — which was tuned for exactly one
configuration — still succeeds often enough to be an expert; the measured
success rate over the range is part of M0's gate rather than an assumption.

Geometry inherited from Lab 8 and why it is not free to change
-------------------------------------------------------------
The pick pedestal stands at y = -0.45 because at y = -0.32 its inner face is
where the right hip passes, and the identical walking controller that manages
twelve steps on bare ground falls on step four (Lab 8 L-M5-f). Its top is at
0.55 m because at 0.72 m the robot walks its own arm into it. This lab widens
the pedestal from 0.10 to 0.14 half-extent to fit two objects, which moves the
inner face from -0.35 to -0.31 — still 0.09 m clear of the hip line, and the
M0 gate re-measures the walk rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from lab9_common import IMAGE_SIZE, OBJECT_NAMES  # noqa: F401  (re-export point)

# Lab 8 (appended to sys.path by lab9_common).
from capstone_scene import (  # noqa: E402
    PAYLOAD_HALF,
    PEDESTAL_TOP,
    PEDESTAL_Y,
)
from g1_torque_model import build_g1_torque_spec  # noqa: E402
from lab8_common import G1_MJCF_PATH, RENDER_HEIGHT, RENDER_WIDTH  # noqa: E402

__all__ = [
    "PICK_X",
    "MARKER_HALF",
    "MARKER_INSET",
    "OBJECT_MASS",
    "PICK_PEDESTAL_HALF",
    "OBJECT_SEPARATION",
    "OBJECT_INSET",
    "Randomisation",
    "VLAScene",
    "build_vla_scene",
]

# Forward position of the pick pedestal, inherited from Lab 8's M5. How many
# steps the robot takes to reach it is decided per episode by which object the
# instruction names — see `expert.approach_steps_for`.
PICK_X: float = 0.40

#: Long in x, narrow in y. Two objects plus a drop marker need length; width is
#: what the robot's hip has to clear as it walks past (Lab 8 L-M5-f: an inner
#: face at y = -0.22 fells a controller that walks twelve steps on bare ground).
#: At half-width 0.13 the inner face sits at -0.32, leaving 0.10 m of clearance.
PICK_PEDESTAL_HALF: tuple[float, float] = (0.22, 0.13)

#: Nominal forward separation between the two objects on the pedestal top, with
#: the drop marker in the gap between them. 0.16 rather than something tighter
#: for two reasons, both measured: the arm has no collision awareness, and at
#: 0.11 m separation the forearm sweeps the distractor off the pedestal on the
#: way to the marker (it moved 0.63 m). And the wider the objects sit, the more
#: the approach differs between them — 1 step versus 4 — which is what makes the
#: walk phase genuinely instruction-dependent.
OBJECT_SEPARATION: float = 0.16

#: The place target: a marked square on the pick pedestal, midway between the
#: two objects. It has to be reachable from wherever the robot stopped, and the
#: stopping point depends on which object was named, so a target fixed to the
#: *world* would be out of reach for one of them. Midway between the objects is
#: within a short reach of either stopping position.
MARKER_HALF: float = 0.045
#: The marker sits further out on the pedestal than the objects do. At the
#: objects' own inset the marker is 0.04 m from the inner edge, and a 60 mm
#: object released accurately on it (9.4 mm) overhangs, tips and slides off —
#: Lab 8 L-M5-j, rediscovered. 0.05 m from the pedestal centre leaves 0.08 m.
MARKER_INSET: float = 0.05
MARKER_RGBA: tuple[float, float, float, float] = (0.95, 0.85, 0.20, 1.0)
#: Both objects sit this far in from the pedestal centre, toward the robot, so
#: the reach stays short even though the prop is held out at y = -0.45. Kept
#: clear of the narrow pedestal's inner edge by more than an object half-extent.
#: 0.09 is Lab 8's value, and it is not free to change: at 0.06 the objects sit
#: 30 mm further out laterally, and *lifting* half a kilogram from there
#: saturates `left_hip_roll` and takes the robot down mid-lift. Lateral reach
#: with mass is the axis with no margin — the same axis that decided Lab 8's
#: gait (L-M3-f) and its walking-reach deferral (L-M4-f).
OBJECT_INSET: float = 0.09

#: Colour families. The hue is jittered inside the family per seed so a policy
#: cannot key on an exact RGB triple, but "red" and "blue" stay unambiguous.
#: Object mass. Lab 8 chose 0.5 kg to make its *carry* test meaningful — an
#: asymmetric load a walking humanoid has to survive. Lab 9 holds the object
#: still at 0.36 m of lateral offset for several seconds while it places, and
#: at 0.5 kg the balance controller diverges there: the DCM error doubles every
#: ~0.15 s (the LIPM rate) with the hand tracking to 5 mm and torques at 21 N.m,
#: which is the signature of a commanded ZMP pinned outside the support polygon
#: rather than of a controller mistuning (tasks/LESSONS.md § L-M0-d).
OBJECT_MASS: float = 0.15

#: The cup's shape. Wide and shallow so it is stable under the tilt an
#: orientation-free hand task leaves it with.
CUP_RADIUS: float = 0.040
CUP_HALF_HEIGHT: float = PAYLOAD_HALF   # same height as the box: the spawn z and
                                        # the place target z are shared constants

OBJECT_RGBA: dict[str, tuple[float, float, float]] = {
    "cup": (0.80, 0.12, 0.12),
    "box": (0.12, 0.25, 0.80),
}

HAND_BODIES: dict[str, str] = {
    "right": "right_wrist_yaw_link",
    "left": "left_wrist_yaw_link",
}

# Head camera aim. The objects sit at z = 0.58 m and the torso camera at
# ~1.21 m, off to the robot's right (y = -0.36). The line of sight to them
# swings from (yaw -48 deg, pitch 53 deg) where the walk starts to (-79, 60) at
# the stopping point, so no fixed aim centres both. A moderate aim plus a wide
# field of view keeps the objects in frame across the whole approach, which is
# what a head camera has to do; centring is not one of its jobs.
HEAD_CAMERA_YAW: float = np.deg2rad(-30.0)
HEAD_CAMERA_PITCH: float = np.deg2rad(40.0)
HEAD_CAMERA_FOVY: float = 90.0

# Wrist camera. The wrist-yaw frame's local +x runs down the arm toward the
# hand tip (measured: world (0.09, -0.20, -0.98) at the stand pose), so the
# camera looks along local +x from slightly up the forearm and offset forward,
# clear of the wrist's own geometry.
WRIST_CAMERA_FOVY: float = 80.0


@dataclass
class Randomisation:
    """Per-seed scene variation, and the ranges it is drawn from.

    Stored alongside every demonstration so a rollout can be reproduced and so
    the evaluation can state the range its success rate was measured over.
    """

    seed: int
    near_object: str          # which object stands closer to the robot
    offsets: dict[str, np.ndarray] = field(default_factory=dict)  # per object, (dx, dy)
    hue_jitter: dict[str, float] = field(default_factory=dict)
    light_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 3.0]))

    #: Sampling ranges. `wide` is the position-randomised evaluation condition
    #: from the brief; training uses `nominal`.
    NOMINAL_XY: tuple[float, float] = (0.020, 0.015)
    WIDE_XY: tuple[float, float] = (0.045, 0.030)
    HUE: float = 0.10

    @classmethod
    def sample(cls, seed: int, wide: bool = False) -> "Randomisation":
        """Draw a scene variation.

        Args:
            seed: Reproducibility seed.
            wide: Use the wider position range (the brief's "position-randomized
                variants"), which is deliberately outside the training range.

        Returns:
            A populated :class:`Randomisation`.
        """
        rng = np.random.default_rng(seed)
        dx, dy = cls.WIDE_XY if wide else cls.NOMINAL_XY
        near = OBJECT_NAMES[int(rng.integers(len(OBJECT_NAMES)))]
        return cls(
            seed=seed,
            near_object=near,
            offsets={
                name: np.array([rng.uniform(-dx, dx), rng.uniform(-dy, dy)])
                for name in OBJECT_NAMES
            },
            hue_jitter={
                name: float(rng.uniform(-cls.HUE, cls.HUE)) for name in OBJECT_NAMES
            },
            light_pos=np.array(
                [rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(2.4, 3.6)]
            ),
        )

    def object_xy(self, name: str) -> np.ndarray:
        """Nominal-plus-jitter world XY for one object.

        Args:
            name: ``"cup"`` or ``"box"``.

        Returns:
            ``(x, y)`` in world coordinates.
        """
        sign = -1.0 if name == self.near_object else 1.0
        base = np.array([PICK_X + sign * OBJECT_SEPARATION, PEDESTAL_Y + OBJECT_INSET])
        return base + self.offsets[name]


@dataclass
class VLAScene:
    """Compiled two-object scene plus the ids the expert and the policy need.

    Presents the same surface as Lab 8's ``CapstoneScene`` — ``payload_position``,
    ``set_weld``, ``pick_position``, ``place_target`` — but resolved against a
    *selected target object*, so Lab 8's phase methods work unmodified while the
    instruction decides which object they act on.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    randomisation: Randomisation
    target: str
    object_bodies: dict[str, int]
    object_qpos: dict[str, int]
    weld_ids: dict[tuple[str, str], int]   # (object, hand) -> equality id
    place_target: np.ndarray
    camera_ids: dict[str, int]

    # -- object access --------------------------------------------------

    def object_half_x(self, name: str) -> float:
        """Half-extent of one object along x [m].

        The grasp point is offset from the object's centre by its own size, so
        a wide cup and a narrow box are approached to the same *surface*
        clearance rather than the same centre distance.

        Args:
            name: ``"cup"`` or ``"box"``.

        Returns:
            Half-extent along x.
        """
        return CUP_RADIUS if name == "cup" else PAYLOAD_HALF

    def object_position(self, name: str) -> np.ndarray:
        """World centre of one object.

        Args:
            name: ``"cup"`` or ``"box"``.

        Returns:
            ``(3,)`` world position.
        """
        return self.data.xpos[self.object_bodies[name]].copy()

    def payload_position(self) -> np.ndarray:
        """World centre of the **target** object (Lab 8 interface)."""
        return self.object_position(self.target)

    @property
    def pick_position(self) -> np.ndarray:
        """Where the target object started (Lab 8 interface)."""
        return self._pick_position.copy()

    def set_target(self, name: str) -> None:
        """Choose which object the manipulation phases act on.

        Args:
            name: ``"cup"`` or ``"box"``.
        """
        if name not in self.object_bodies:
            raise ValueError(f"unknown object {name!r}")
        self.target = name
        self._pick_position = self.object_position(name)

    # -- welds ----------------------------------------------------------

    def _ids(self, which: str) -> tuple[int, ...]:
        if which == "both":
            hands = ("right", "left")
        elif which in ("right", "left"):
            hands = (which,)
        else:
            raise ValueError(f"which must be right/left/both, got {which!r}")
        return tuple(self.weld_ids[(self.target, hand)] for hand in hands)

    def set_weld(self, active: bool, which: str = "right") -> None:
        """Open or close a grasp weld on the *target* object.

        Closing captures the live hand->object transform first. MuJoCo's weld
        holds its **compile-time** relative pose, so activating it naively
        commands a snap back to the rest configuration rather than a grasp —
        Lab 8 measured a 0.42 m lurch that threw the robot down (L-M5-b).

        Args:
            active: Close (True) or open (False).
            which: ``"right"``, ``"left"`` or ``"both"``.
        """
        for weld_id in self._ids(which):
            if active:
                self._capture_relative_pose(weld_id)
            self.data.eq_active[weld_id] = int(active)
        mujoco.mj_forward(self.model, self.data)

    def _capture_relative_pose(self, weld_id: int) -> None:
        """Write the live hand->object transform into ``eq_data[3:10]``.

        Layout for ``mjEQ_WELD``: anchor(3), relpose(7: pos + wxyz quat),
        torquescale(1). ``relpose`` is body2 expressed in body1.
        """
        hand = self.model.eq_obj1id[weld_id]
        obj = self.model.eq_obj2id[weld_id]

        hand_rot = self.data.xmat[hand].reshape(3, 3)
        relative_pos = hand_rot.T @ (self.data.xpos[obj] - self.data.xpos[hand])

        hand_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(hand_quat_inv, self.data.xquat[hand])
        relative_quat = np.zeros(4)
        mujoco.mju_mulQuat(relative_quat, hand_quat_inv, self.data.xquat[obj])

        self.model.eq_data[weld_id, 0:3] = 0.0
        self.model.eq_data[weld_id, 3:6] = relative_pos
        self.model.eq_data[weld_id, 6:10] = relative_quat
        self.model.eq_data[weld_id, 10] = 1.0

    def any_weld_active(self) -> bool:
        """True if any grasp weld in the scene is closed."""
        return bool(np.any([self.data.eq_active[i] for i in self.weld_ids.values()]))


def _camera_quat(right: np.ndarray, up: np.ndarray) -> list[float]:
    """Camera orientation quaternion (w, x, y, z) from right and up axes.

    `MjsCamera` exposes `quat`, not the MJCF `xyaxes` shortcut, so the frame is
    assembled explicitly. A MuJoCo camera looks along its own **-z** with +y up,
    so the frame's columns are (right, up, -forward).

    Args:
        right: Camera-x in the parent body's frame.
        up: Camera-y in the parent body's frame.

    Returns:
        Four floats, ``(w, x, y, z)``.
    """
    x = np.asarray(right, dtype=float)
    x /= np.linalg.norm(x)
    y = np.asarray(up, dtype=float)
    y -= x * (x @ y)          # re-orthogonalise rather than trust the caller
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).reshape(9))
    return quat.tolist()


def _look_direction_camera(yaw: float, pitch: float) -> list[float]:
    """Quaternion for a camera aimed by yaw about +z and pitch below horizontal.

    Args:
        yaw: Rotation about the parent's +z, positive toward +y [rad].
        pitch: Downward pitch from horizontal [rad].

    Returns:
        Four floats, ``(w, x, y, z)``.
    """
    forward = np.array(
        [np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)]
    )
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    return _camera_quat(right=right, up=np.cross(right, forward))


def build_vla_scene(
    timestep: float,
    randomisation: Randomisation,
    target: str = "cup",
) -> VLAScene:
    """Compile the torque-actuated G1 with two objects, pedestals and cameras.

    Args:
        timestep: Simulation timestep [s].
        randomisation: Per-seed scene variation.
        target: Which object the manipulation phases act on initially.

    Returns:
        A :class:`VLAScene`.

    Both objects carry freejoints, so every pose a gate asserts on is a genuine
    simulation outcome rather than a commanded value (Lab 5's lesson).
    """
    spec = build_g1_torque_spec(G1_MJCF_PATH, with_floor=True, timestep=timestep)

    def add_pedestal(name: str, x: float, y: float,
                     half: tuple[float, float]) -> None:
        body = spec.worldbody.add_body()
        body.name = name
        body.pos = [x, y, 0.0]
        geom = body.add_geom()
        geom.name = f"{name}_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [half[0], half[1], PEDESTAL_TOP / 2.0]
        geom.pos = [0.0, 0.0, PEDESTAL_TOP / 2.0]
        geom.rgba = [0.35, 0.35, 0.40, 1.0]
        geom.condim = 3
        geom.friction = [0.9, 0.005, 0.0001]
        # Zero mass: static scenery welded to the world, and a density-derived
        # mass would skew any sum over body masses (Lab 8's note).
        geom.mass = 0.0

    add_pedestal("pick_pedestal", PICK_X, PEDESTAL_Y, PICK_PEDESTAL_HALF)

    # The drop marker. The `place` task it was built for did not survive M0
    # (L-M0-e), but the marker stays: `approach_steps_for` still stops the robot
    # at the midpoint of object and marker, which is what keeps the near and far
    # objects at comparable reach, and a visible landmark between the two
    # objects gives the head camera something scale-bearing to see. It carries
    # no mass and no collision, so it cannot perturb anything.
    marker = spec.worldbody.add_body()
    marker.name = "drop_marker"
    marker.pos = [PICK_X, PEDESTAL_Y + MARKER_INSET, PEDESTAL_TOP + 0.001]
    plate = marker.add_geom()
    plate.name = "drop_marker_geom"
    plate.type = mujoco.mjtGeom.mjGEOM_BOX
    plate.size = [MARKER_HALF, MARKER_HALF, 0.001]
    plate.rgba = list(MARKER_RGBA)
    plate.contype = 0
    plate.conaffinity = 0
    plate.mass = 0.0

    for name in OBJECT_NAMES:
        xy = randomisation.object_xy(name)
        body = spec.worldbody.add_body()
        body.name = name
        body.pos = [float(xy[0]), float(xy[1]), PEDESTAL_TOP + PAYLOAD_HALF]
        body.add_freejoint()
        geom = body.add_geom()
        geom.name = f"{name}_geom"
        if name == "cup":
            # A cylinder of the same half-extent and mass as Lab 8's payload,
            # so the inertia the controller is told about at grasp time (a box
            # approximation, `attach_payload_to_pinocchio`) stays as good an
            # approximation for one object as for the other.
            # Short and wide, not a tall cylinder. The hand tasks control
            # position only, so the object's orientation at release is whatever
            # the wrist happened to be holding; a tall cylinder released with a
            # few degrees of tilt topples and rolls off the marker, which showed
            # up as a systematic 58-73 mm placement error on the cup while the
            # box placed to 25-40 mm from identical hand tracking.
            geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            geom.size = [CUP_RADIUS, CUP_HALF_HEIGHT, 0.0]
        else:
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = [PAYLOAD_HALF] * 3
        base = np.array(OBJECT_RGBA[name])
        jitter = randomisation.hue_jitter[name]
        rgb = np.clip(base + jitter * np.array([1.0, 0.3, 0.3]), 0.03, 0.97)
        geom.rgba = [*rgb.tolist(), 1.0]
        geom.mass = OBJECT_MASS
        geom.condim = 3
        geom.friction = [0.9, 0.005, 0.0001]

    # -- welds: every (object, hand) pair, all inactive -------------------
    for name in OBJECT_NAMES:
        for hand, hand_body in HAND_BODIES.items():
            weld = spec.add_equality()
            weld.name = f"grasp_{name}_{hand}"
            weld.type = mujoco.mjtEq.mjEQ_WELD
            weld.objtype = mujoco.mjtObj.mjOBJ_BODY
            weld.name1 = hand_body
            weld.name2 = name
            weld.active = False
            # Lab 8's compliant grasp: solved rigidly the weld transmits the
            # payload's inertial reaction to the wrist as a near-impulse at
            # every contact switch.
            weld.solref = [0.02, 1.0]
            weld.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]

    # -- egocentric cameras ----------------------------------------------
    torso = spec.body("torso_link")
    head_cam = torso.add_camera()
    head_cam.name = "head"
    head_cam.pos = [0.08, 0.0, 0.38]
    head_cam.quat = _look_direction_camera(HEAD_CAMERA_YAW, HEAD_CAMERA_PITCH)
    head_cam.fovy = HEAD_CAMERA_FOVY

    wrist = spec.body(HAND_BODIES["right"])
    wrist_cam = wrist.add_camera()
    wrist_cam.name = "wrist"
    wrist_cam.pos = [-0.03, 0.0, 0.07]
    # Looks along the wrist frame's local +x — down the arm, past the hand tip.
    wrist_cam.quat = _camera_quat(
        right=np.array([0.0, -1.0, 0.0]), up=np.array([0.0, 0.0, 1.0])
    )
    wrist_cam.fovy = WRIST_CAMERA_FOVY

    # Randomised key light. The default Menagerie light stays as fill.
    light = spec.worldbody.add_light()
    light.name = "key_light"
    light.pos = randomisation.light_pos.tolist()
    light.dir = [0.0, 0.0, -1.0]
    light.type = mujoco.mjtLightType.mjLIGHT_SPOT
    light.castshadow = False   # shadows cost 4x per frame in software rendering

    model = spec.compile()
    model.vis.global_.offwidth = max(RENDER_WIDTH, IMAGE_SIZE)
    model.vis.global_.offheight = max(RENDER_HEIGHT, IMAGE_SIZE)
    data = mujoco.MjData(model)

    object_bodies = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in OBJECT_NAMES
    }
    if any(idx < 0 for idx in object_bodies.values()):
        raise RuntimeError("VLA scene did not compile both objects")
    object_qpos = {
        name: int(model.jnt_qposadr[model.body_jntadr[idx]])
        for name, idx in object_bodies.items()
    }

    if model.nkey > 0:
        # Menagerie's keyframe was authored for the robot alone; recompiling
        # with two freejoint objects lengthens qpos and the keyframe is
        # zero-padded, which drops both objects through the floor with a
        # (0,0,0,0) quaternion. Restore the scene's own defaults from qpos0.
        mujoco.mj_resetDataKeyframe(model, data, 0)
        first_scene_qpos = min(object_qpos.values())
        data.qpos[first_scene_qpos:] = model.qpos0[first_scene_qpos:]

    weld_ids = {}
    for name in OBJECT_NAMES:
        for hand in HAND_BODIES:
            weld_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{name}_{hand}"
            )
            if weld_id < 0:
                raise RuntimeError(f"weld grasp_{name}_{hand} did not compile")
            weld_ids[(name, hand)] = weld_id
            data.eq_active[weld_id] = 0

    camera_ids = {}
    for cam in ("head", "wrist"):
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        if cam_id < 0:
            raise RuntimeError(f"camera {cam!r} did not compile")
        camera_ids[cam] = cam_id

    mujoco.mj_forward(model, data)

    scene = VLAScene(
        model=model,
        data=data,
        randomisation=randomisation,
        target=target,
        object_bodies=object_bodies,
        object_qpos=object_qpos,
        weld_ids=weld_ids,
        place_target=np.array(
            [PICK_X, PEDESTAL_Y + MARKER_INSET, PEDESTAL_TOP + PAYLOAD_HALF],
            dtype=float,
        ),
        camera_ids=camera_ids,
    )
    scene.set_target(target)
    return scene
