"""Lab 8 — M5: the capstone scene (pedestals, payload, weld grasp).

Adds to the torque-actuated G1 everything the loco-manipulation sequence needs
and nothing it does not:

* a **pick pedestal** with a payload box standing on it, placed so the robot
  arrives in front of it after M3's twelve-step walk;
* a **place pedestal** further along the same line;
* **two weld** equality constraints, right wrist and left wrist to the payload,
  both built inactive — the "grasp stays SIMPLE" instruction in
  `plan/LAB_08.md`, and the same mechanism Lab 6 used for its cooperative
  carry. The right weld closes at the pick; the left joins once the load has
  been brought to the chest, because that is the only place the left hand can
  reach it (the pedestal stands off to the right of the walking line) and
  because a two-handed hold is the configuration M4 showed this robot can
  actually walk with (L-M4-f, L-M5-g).

Why a weld and not fingers
--------------------------
The Menagerie G1 in this lab has no hand — the kinematic chain ends at
`*_wrist_yaw_link`. Lab 5 already built and validated a real parallel-jaw
grasp on the UR5e, and repeating that here would be re-litigating a solved
problem with a model that cannot express it. The brief asks for the *loco*-
manipulation result: whether the whole-body controller survives picking mass
up, carrying it, and putting it down. A weld isolates exactly that question.

What the payload is allowed to be
---------------------------------
0.5 kg on a 33.34 kg robot, held out to one side — the brief's "40 mm
cube-class object". Small, but not negligible: it arrives as a *step change*
in the centroidal model at the instant of the weld, and it is carried
asymmetrically, which M4 established is the hard direction for this robot
(L-M4-f). M5 replans after the grasp rather than hoping the controller absorbs
the change (L-M4-a).

A first attempt at 1.5 kg / 90 mm is recorded in L-M5-d: the robot stands with
it without trouble and falls on the carry leg every single time. Walking with
asymmetric mass, not lifting it, is what costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from g1_torque_model import build_g1_torque_spec
from lab8_common import G1_MJCF_PATH, RENDER_HEIGHT, RENDER_WIDTH

__all__ = [
    "PAYLOAD_BODY",
    "PAYLOAD_MASS",
    "PAYLOAD_HALF",
    "WELD_NAME",
    "WELD_NAME_LEFT",
    "CapstoneScene",
    "build_capstone_scene",
]

PAYLOAD_BODY = "payload"
WELD_NAME = "grasp_weld"            # right wrist — the picking hand
WELD_NAME_LEFT = "grasp_weld_left"  # left wrist — joins for the carry

# `plan/LAB_08.md` M5 says "reuse Lab 5 sizing: 40 mm cube-class object". The
# first attempt used a 90 mm, 1.5 kg block instead, and it does not walk: the
# robot stands with it happily and falls on the carry leg every time, at every
# hand weight and with the hand task off entirely (L-M5-d). 0.5 kg held out to
# one side is already a real asymmetric load on a 33 kg biped.
PAYLOAD_MASS = 0.5          # kg — 1.5 % of the G1's 33.34 kg
PAYLOAD_HALF = 0.030        # half-extent of the cube [m] → a 60 mm cube

PEDESTAL_HALF_XY = 0.10
# Pedestal height is a **controller** parameter, not set dressing. At 0.72 m
# the top sits exactly where the G1's wrists hang while walking (y ≈ −0.22,
# z ≈ 0.72), and the robot walks its own arm into the pick pedestal on the way
# past: the identical M3 controller walks 12 steps on the bare model and falls
# on step 4 of the capstone scene, at x ≈ 0.41 — the pedestal's position
# (L-M5-f). Dropping to 0.55 m puts the whole prop below the arm swing and
# turns the reach into a natural down-and-forward motion.
PEDESTAL_TOP = 0.55
# Both pedestals stand **beside** the walking line, on the working arm's side.
# The gait only travels forward, so a pedestal on the line would either be
# walked into after the pick or force the place target behind the robot. A
# lateral offset removes the collision entirely and costs nothing: the right
# hand rests at y ≈ −0.22 already, so reaching y = −0.32 is a small motion.
# Far enough out that the robot's own hip clears it. The first attempt put the
# pedestal centre at y = −0.32, whose inner face at −0.22 is exactly where the
# right hip passes: the contact log showed `pick_pedestal ↔ right_hip_roll_link`
# at t = 3.47 s, then the ankle, then a fall — the identical M3 controller that
# walks 12 steps on the bare model (L-M5-f). The payload is then set on the
# prop's *inner* edge so the reach stays short even though the prop is far.
PEDESTAL_Y = -0.45
PAYLOAD_INSET = 0.09        # pick: payload offset from the pedestal centre, toward the robot
# The place target sits closer to its pedestal's centre. At the pick inset the
# 30 mm box overhangs a 0.10 m half-extent by 20 mm: it is released accurately
# (18.9 mm from target) and then slides off the inner edge and drops to the
# floor. Widening the pedestal instead puts it inside the robot's own standing
# space — the same collision class as L-M5-f, and it fell with the torque
# saturated. The target moves; the furniture stays out of the way (L-M5-j).
PLACE_INSET = 0.05


@dataclass
class CapstoneScene:
    """Compiled capstone model plus the ids the sequence needs."""

    model: mujoco.MjModel
    data: mujoco.MjData
    payload_body: int
    payload_qpos: int          # index of the payload freejoint in qpos
    weld_id: int               # right wrist
    weld_left_id: int          # left wrist
    pick_position: np.ndarray  # payload centre at t=0 (world)
    place_target: np.ndarray   # where the payload must end up (world)

    def payload_position(self) -> np.ndarray:
        """Current payload centre in world coordinates."""
        return self.data.xpos[self.payload_body].copy()

    def _weld_ids(self, which: str) -> tuple[int, ...]:
        if which == "right":
            return (self.weld_id,)
        if which == "left":
            return (self.weld_left_id,)
        if which == "both":
            return (self.weld_id, self.weld_left_id)
        raise ValueError(f"which must be right/left/both, got '{which}'")

    def set_weld(self, active: bool, which: str = "right") -> None:
        """Enable or disable the grasp weld.

        `eq_active` is the runtime switch; the constraint itself is compiled in
        so the model's dimensions never change mid-episode — the QP is already
        rebuilt on every contact switch and does not need a second source of
        resizing.

        Closing the weld **captures the current relative pose first**. MuJoCo's
        weld holds body2 at `eq_data`'s relpose relative to body1, and that
        field is baked at compile time — from the rest pose, where the hand is
        at x = −0.02 and the payload is at x = 0.40. Activating without
        refreshing it does not grasp the payload; it commands a 0.42 m snap,
        which is exactly what the simulator delivered: the payload leapt
        0.115 m and took the robot down with it (L-M5-b).
        """
        for weld_id in self._weld_ids(which):
            if active:
                self._capture_relative_pose(weld_id)
            self.data.eq_active[weld_id] = int(active)
        mujoco.mj_forward(self.model, self.data)

    def _capture_relative_pose(self, weld_id: int) -> None:
        """Write the live hand→payload transform into the weld's `eq_data`.

        Layout for `mjEQ_WELD` is `anchor(3), relpose(7: pos + wxyz quat),
        torquescale(1)`. `relpose` is body2 (payload) expressed in body1
        (hand); the anchor is the weld point in body2's frame, which for a
        grasp is the payload's own origin.
        """
        hand = self.model.eq_obj1id[weld_id]
        payload = self.model.eq_obj2id[weld_id]

        hand_rot = self.data.xmat[hand].reshape(3, 3)
        relative_pos = hand_rot.T @ (self.data.xpos[payload] - self.data.xpos[hand])

        hand_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(hand_quat_inv, self.data.xquat[hand])
        relative_quat = np.zeros(4)
        mujoco.mju_mulQuat(relative_quat, hand_quat_inv, self.data.xquat[payload])

        self.model.eq_data[weld_id, 0:3] = 0.0
        self.model.eq_data[weld_id, 3:6] = relative_pos
        self.model.eq_data[weld_id, 6:10] = relative_quat
        self.model.eq_data[weld_id, 10] = 1.0


def build_capstone_scene(
    timestep: float,
    pick_x: float,
    place_x: float,
    hand_body: str = "right_wrist_yaw_link",
    left_hand_body: str = "left_wrist_yaw_link",
    pedestal_y: float = PEDESTAL_Y,
    place_pedestal_y: float | None = None,
) -> CapstoneScene:
    """Compile the G1 torque model with pedestals, payload and a grasp weld.

    Args:
        timestep: Simulation timestep [s].
        pick_x: Forward position of the pick pedestal [m].
        place_x: Forward position of the place pedestal [m].
        hand_body: Body the payload welds to at grasp time.

    The pedestals are `mjGEOM_BOX` statics with the robot's floor friction, and
    the payload carries a freejoint so its pose is a genuine simulation
    outcome — the gate asserts on `data.xpos`, never on a commanded value
    (Lab 5's lesson: DONE must verify the object actually moved).
    """
    spec = build_g1_torque_spec(G1_MJCF_PATH, with_floor=True, timestep=timestep)

    def add_pedestal(name: str, x: float, top: float, y: float,
                     half_xy: float = PEDESTAL_HALF_XY) -> None:
        body = spec.worldbody.add_body()
        body.name = name
        body.pos = [x, y, 0.0]
        geom = body.add_geom()
        geom.name = f"{name}_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [half_xy, half_xy, top / 2.0]
        geom.pos = [0.0, 0.0, top / 2.0]
        geom.rgba = [0.35, 0.35, 0.40, 1.0]
        geom.condim = 3
        geom.friction = [0.9, 0.005, 0.0001]
        # Static scenery. Zero mass keeps it out of every CoM/mass sum in the
        # model — the pedestals are welded to the world and contribute nothing
        # to the dynamics, but MuJoCo would still give them density-derived
        # mass and skew any `sum(body_mass)` or world-subtree CoM read.
        geom.mass = 0.0

    # The place pedestal may stand closer to the walking line than the pick
    # one: the robot stops in front of it and never walks past, so the hip
    # clearance that forces the pick pedestal out to y = −0.45 does not apply.
    # Closer is better — the place reach is the longest one in the sequence,
    # made from a carry pose rather than from rest.
    place_y = pedestal_y if place_pedestal_y is None else place_pedestal_y
    add_pedestal("pick_pedestal", pick_x, PEDESTAL_TOP, pedestal_y)
    add_pedestal("place_pedestal", place_x, PEDESTAL_TOP, place_y)

    payload = spec.worldbody.add_body()
    payload.name = PAYLOAD_BODY
    payload.pos = [pick_x, pedestal_y + PAYLOAD_INSET, PEDESTAL_TOP + PAYLOAD_HALF]
    payload.add_freejoint()
    box = payload.add_geom()
    box.name = "payload_geom"
    box.type = mujoco.mjtGeom.mjGEOM_BOX
    box.size = [PAYLOAD_HALF] * 3
    box.rgba = [0.85, 0.45, 0.15, 1.0]
    box.mass = PAYLOAD_MASS
    box.condim = 3
    box.friction = [0.9, 0.005, 0.0001]

    for name, body in ((WELD_NAME, hand_body), (WELD_NAME_LEFT, left_hand_body)):
        weld = spec.add_equality()
        weld.name = name
        weld.type = mujoco.mjtEq.mjEQ_WELD
        weld.objtype = mujoco.mjtObj.mjOBJ_BODY
        weld.name1 = body
        weld.name2 = PAYLOAD_BODY
        weld.active = False
        # A stiff-but-not-rigid weld. Solved rigidly, the grasp transmits the
        # payload's inertial reaction to the wrist as a near-impulse at contact
        # switches; a short solref lets it behave like a firm hand instead.
        weld.solref = [0.02, 1.0]
        weld.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]

    model = spec.compile()
    model.vis.global_.offwidth = RENDER_WIDTH
    model.vis.global_.offheight = RENDER_HEIGHT
    data = mujoco.MjData(model)
    if model.nkey > 0:
        # Menagerie's keyframe was authored for the robot alone. Recompiling
        # with the payload lengthens qpos, and the keyframe is zero-padded —
        # which drops the payload through the floor with a (0,0,0,0)
        # quaternion. Restore the scene's own defaults from `qpos0`.
        mujoco.mj_resetDataKeyframe(model, data, 0)
        scene_qpos = int(model.jnt_qposadr[
            model.body_jntadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PAYLOAD_BODY)
            ]
        ])
        data.qpos[scene_qpos:] = model.qpos0[scene_qpos:]

    payload_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PAYLOAD_BODY)
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, WELD_NAME)
    weld_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, WELD_NAME_LEFT)
    if payload_body < 0 or weld_id < 0 or weld_left_id < 0:
        raise RuntimeError("capstone scene did not compile the payload or its weld")
    payload_qpos = int(model.jnt_qposadr[model.body_jntadr[payload_body]])

    data.eq_active[weld_id] = 0
    data.eq_active[weld_left_id] = 0
    mujoco.mj_forward(model, data)

    return CapstoneScene(
        model=model,
        data=data,
        payload_body=payload_body,
        payload_qpos=payload_qpos,
        weld_id=weld_id,
        weld_left_id=weld_left_id,
        pick_position=data.xpos[payload_body].copy(),
        place_target=np.array(
            [place_x, place_y + PLACE_INSET, PEDESTAL_TOP + PAYLOAD_HALF],
            dtype=float,
        ),
    )
