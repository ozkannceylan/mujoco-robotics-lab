"""
record_pro_demo.py — Professional pick-and-place demo video for Lab 5.

Uses MuJoCo Menagerie UR5e + Robotiq 2F-85 gripper with real mesh visuals.
Outputs: ../media/pick_place_pro.mp4 (1280x720, 60 fps)

Architecture:
    Pinocchio = NOT used here (standalone MuJoCo IK via Jacobian DLS)
    MuJoCo   = simulation + rendering + video recording
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import imageio
import imageio_ffmpeg
import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
MENAGERIE = REPO_ROOT / "lab-2-Ur5e-robotics-lab" / "models" / "mujoco_menagerie"
UR5E_XML = MENAGERIE / "universal_robots_ur5e" / "ur5e.xml"
ROBOTIQ_XML = MENAGERIE / "robotiq_2f85" / "2f85.xml"
MEDIA_DIR = Path(__file__).resolve().parents[1] / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO = MEDIA_DIR / "pick_place_pro.mp4"

# Lab 4 collision-free planning
_LAB4_SRC = REPO_ROOT / "lab-4-motion-planning" / "src"
_LAB4_URDF = REPO_ROOT / "lab-4-motion-planning" / "models" / "ur5e_collision.urdf"
if str(_LAB4_SRC) not in sys.path:
    sys.path.insert(0, str(_LAB4_SRC))

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------
TABLE_POS = np.array([0.45, 0.0, 0.30])          # table centre
TABLE_SIZE = np.array([0.30, 0.40, 0.015])        # half-extents
TABLE_SURFACE_Z = TABLE_POS[2] + TABLE_SIZE[2]    # z = 0.315

BOX_HALF = 0.025                                   # 5 cm cube
BOX_A_POS = np.array([0.35,  0.20, TABLE_SURFACE_Z + BOX_HALF])   # [0.35,  0.20, 0.340]
BOX_B_POS = np.array([0.35, -0.20, TABLE_SURFACE_Z + BOX_HALF])   # [0.35, -0.20, 0.340]

# Gripper tip offset from attachment_site (verified at Q_HOME = 0.1558 m)
GRIPPER_TIP_OFFSET = 0.1558

# Hover height above object
HOVER_HEIGHT = 0.15

Q_HOME = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

# Position servo gains (from Menagerie gainprm)
KP_VEC = np.array([2000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0])


# ---------------------------------------------------------------------------
# Collision-free planning (Lab 4 RRT*)
# ---------------------------------------------------------------------------

from collision_checker import CollisionChecker  # noqa: E402 (after sys.path insert)
from rrt_planner import RRTStarPlanner          # noqa: E402
from trajectory_smoother import shortcut_path   # noqa: E402

# Collision checker: arm self-collision + table only.
# obstacle_specs=() disables the Lab 4 scene obstacles (obs1-obs4) which do not
# exist in the pro-demo scene; include_table=True adds the table separately.
_cc = CollisionChecker(obstacle_specs=(), self_collision=True, include_table=True)
_planner = RRTStarPlanner(
    _cc,
    step_size=0.3,
    goal_bias=0.15,
    rewire_radius=1.0,
    goal_tolerance=0.15,
)


def plan_collision_free(
    q_start: np.ndarray,
    q_end: np.ndarray,
    label: str = "",
    max_iter: int = 6000,
    seed: int = 42,
) -> list[np.ndarray]:
    """Plan a collision-free joint-space path via RRT* + shortcutting.

    Args:
        q_start: Start configuration (6,).
        q_end: Goal configuration (6,).
        label: Human-readable label for error messages.
        max_iter: RRT* iteration budget.
        seed: Random seed for reproducibility.

    Returns:
        List of collision-free waypoints from q_start to q_end.

    Raises:
        RuntimeError: If RRT* fails to find a path.
    """
    path = _planner.plan(q_start, q_end, max_iter=max_iter, rrt_star=True, seed=seed)
    if path is None:
        raise RuntimeError(
            f"RRT* failed to find a collision-free path "
            f"{'for ' + label + ' ' if label else ''}(q_start={np.round(q_start, 2)}, "
            f"q_end={np.round(q_end, 2)}). "
            "Check joint limits and obstacle placement."
        )
    path = shortcut_path(path, _cc, max_iter=200, seed=seed)
    return path


# ---------------------------------------------------------------------------
# Build combined MjSpec scene
# ---------------------------------------------------------------------------

def build_scene() -> mujoco.MjSpec:
    """Combine UR5e + gripper, then add table, box, floor, lights."""
    robot: mujoco.MjSpec = mujoco.MjSpec.from_file(str(UR5E_XML))
    gripper: mujoco.MjSpec = mujoco.MjSpec.from_file(str(ROBOTIQ_XML))

    attach_site = next(s for s in robot.sites if s.name == "attachment_site")
    robot.attach(gripper, prefix="2f85_", site=attach_site)

    # ------------------------------------------------------------------
    # Visual quality settings
    # ------------------------------------------------------------------
    robot.stat.center = [0.3, 0.0, 0.4]
    robot.stat.extent = 1.2

    robot.visual.headlight.active = True
    robot.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    robot.visual.headlight.ambient = [0.3, 0.3, 0.3]
    robot.visual.headlight.specular = [0.2, 0.2, 0.2]

    # Set offscreen framebuffer to support 1280x720
    robot.visual.global_.offwidth = 1280
    robot.visual.global_.offheight = 720

    # ------------------------------------------------------------------
    # Textures
    # ------------------------------------------------------------------
    sky_tex = robot.add_texture()
    sky_tex.name = "sky_tex"
    sky_tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky_tex.rgb1 = [0.45, 0.65, 0.85]
    sky_tex.rgb2 = [0.05, 0.10, 0.30]
    sky_tex.width = 512
    sky_tex.height = 512

    floor_tex = robot.add_texture()
    floor_tex.name = "floor_tex"
    floor_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    floor_tex.rgb1 = [0.90, 0.90, 0.90]
    floor_tex.rgb2 = [0.55, 0.55, 0.55]
    floor_tex.width = 512
    floor_tex.height = 512

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------
    floor_mat = robot.add_material()
    floor_mat.name = "floor_mat"
    floor_mat.textures = ["floor_tex"] + [""] * 9
    floor_mat.texrepeat = [6.0, 6.0]
    floor_mat.specular = 0.1
    floor_mat.shininess = 0.1

    table_mat = robot.add_material()
    table_mat.name = "table_mat"
    table_mat.rgba = [0.55, 0.38, 0.20, 1.0]
    table_mat.specular = 0.15
    table_mat.shininess = 0.2

    box_mat = robot.add_material()
    box_mat.name = "box_mat"
    box_mat.rgba = [0.20, 0.55, 0.90, 1.0]
    box_mat.specular = 0.4
    box_mat.shininess = 0.5

    target_mat = robot.add_material()
    target_mat.name = "target_mat"
    target_mat.rgba = [0.20, 0.80, 0.20, 0.55]
    target_mat.specular = 0.0

    table_leg_mat = robot.add_material()
    table_leg_mat.name = "table_leg_mat"
    table_leg_mat.rgba = [0.40, 0.40, 0.40, 1.0]
    table_leg_mat.specular = 0.3

    # ------------------------------------------------------------------
    # World-body scene elements
    # ------------------------------------------------------------------
    wb = robot.worldbody

    # Floor
    floor_g = wb.add_geom()
    floor_g.name = "floor"
    floor_g.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor_g.size = [0, 0, 0.05]
    floor_g.material = "floor_mat"

    # Table top
    table_body = wb.add_body()
    table_body.name = "table"
    table_body.pos = TABLE_POS.tolist()
    top_g = table_body.add_geom()
    top_g.name = "table_top"
    top_g.type = mujoco.mjtGeom.mjGEOM_BOX
    top_g.size = TABLE_SIZE.tolist()
    top_g.material = "table_mat"

    # Table legs (4 legs)
    leg_half = [0.02, 0.02, TABLE_POS[2] / 2]
    leg_offsets = [
        ( 0.25,  0.35, -TABLE_POS[2] / 2 - TABLE_SIZE[2]),
        (-0.25,  0.35, -TABLE_POS[2] / 2 - TABLE_SIZE[2]),
        ( 0.25, -0.35, -TABLE_POS[2] / 2 - TABLE_SIZE[2]),
        (-0.25, -0.35, -TABLE_POS[2] / 2 - TABLE_SIZE[2]),
    ]
    for i, (lx, ly, lz) in enumerate(leg_offsets):
        leg_g = table_body.add_geom()
        leg_g.name = f"table_leg_{i}"
        leg_g.type = mujoco.mjtGeom.mjGEOM_BOX
        leg_g.size = leg_half
        leg_g.pos = [lx, ly, lz]
        leg_g.material = "table_leg_mat"

    # Pick box (body A) — has a freejoint so physics applies
    box_body = wb.add_body()
    box_body.name = "box_A"
    box_body.pos = BOX_A_POS.tolist()
    box_fj = box_body.add_freejoint()
    box_fj.name = "box_joint"
    box_g = box_body.add_geom()
    box_g.name = "box_A_geom"
    box_g.type = mujoco.mjtGeom.mjGEOM_BOX
    box_g.size = [BOX_HALF, BOX_HALF, BOX_HALF]
    box_g.material = "box_mat"
    box_g.mass = 0.1

    # Target zone marker at BOX_B (static visual only)
    target_body = wb.add_body()
    target_body.name = "target_zone"
    target_body.pos = BOX_B_POS.tolist()
    tg = target_body.add_geom()
    tg.name = "target_geom"
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = [BOX_HALF + 0.003, BOX_HALF + 0.003, 0.003]
    tg.pos = [0, 0, -BOX_HALF - 0.003]
    tg.material = "target_mat"
    tg.contype = 0
    tg.conaffinity = 0

    # Lights — 3-point lighting setup
    key_light = wb.add_light()
    key_light.name = "key_light"
    key_light.pos = [1.5, 1.0, 2.5]
    key_light.dir = [-0.6, -0.4, -1.0]
    key_light.diffuse = [0.8, 0.8, 0.8]
    key_light.specular = [0.3, 0.3, 0.3]
    key_light.castshadow = True

    fill_light = wb.add_light()
    fill_light.name = "fill_light"
    fill_light.pos = [-1.5, -0.5, 2.0]
    fill_light.dir = [0.6, 0.2, -1.0]
    fill_light.diffuse = [0.4, 0.4, 0.5]
    fill_light.specular = [0.05, 0.05, 0.05]
    fill_light.castshadow = False

    back_light = wb.add_light()
    back_light.name = "back_light"
    back_light.pos = [0.0, -2.0, 2.5]
    back_light.dir = [0.0, 0.8, -1.0]
    back_light.diffuse = [0.3, 0.3, 0.3]
    back_light.specular = [0.1, 0.1, 0.1]
    back_light.castshadow = False

    return robot


# ---------------------------------------------------------------------------
# SO(3) utilities
# ---------------------------------------------------------------------------

def _so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm map: rotation matrix → rotation vector.

    Returns the axis-angle vector v such that exp([v]×) = R.
    Handles the 180° case correctly (no singularity at θ = π).

    Args:
        R: 3×3 rotation matrix.

    Returns:
        Rotation vector (3,) in radians.
    """
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    if abs(theta) < 1e-8:
        return np.zeros(3)
    return (theta / (2.0 * np.sin(theta))) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


# ---------------------------------------------------------------------------
# DLS Jacobian IK
# ---------------------------------------------------------------------------

def compute_ik(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    q_init: np.ndarray,
    max_iter: int = 800,
    tol: float = 1e-4,
    alpha: float = 0.3,
    lambda_sq: float = 1e-4,
) -> np.ndarray:
    """Damped Least-Squares IK using MuJoCo Jacobian (6-DOF arm only).

    Constrains joint angles to [-pi, pi] via wrapping to avoid winding
    solutions that are numerically valid but physically far from q_init.
    """
    q = q_init.copy()
    best_q = q.copy()
    best_err = np.inf
    for _ in range(max_iter):
        d.qpos[:6] = q
        mujoco.mj_forward(m, d)

        pos_err = target_pos - d.site_xpos[site_id]
        R_cur = d.site_xmat[site_id].reshape(3, 3)
        # SO(3) log: singularity-free at all rotation magnitudes including 180°.
        # target_rot @ R_cur.T = relative rotation from current to target.
        ori_err = _so3_log(target_rot @ R_cur.T)
        err = np.concatenate([pos_err, ori_err])
        err_norm = np.linalg.norm(err)

        if err_norm < best_err:
            best_err = err_norm
            best_q = q.copy()

        if err_norm < tol:
            return q

        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :6], jacr[:, :6]])
        JJt = J @ J.T + lambda_sq * np.eye(6)
        dq = alpha * J.T @ np.linalg.solve(JJt, err)
        q = q + dq
        # Soft-clip to ±2π (actual UR5e hardware limit) — avoid hard modular wrapping
        # which causes discontinuous joint values and "spinning" trajectories.
        q = np.clip(q, -2 * np.pi, 2 * np.pi)
    return best_q


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def make_renderer(m: mujoco.MjModel, width: int = 1280, height: int = 720) -> mujoco.Renderer:
    """Create and configure an off-screen renderer."""
    renderer = mujoco.Renderer(m, height=height, width=width)
    return renderer


def setup_camera(renderer: mujoco.Renderer) -> None:
    """Configure the camera for a nice 3/4 view of the scene."""
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True


def render_frame(
    renderer: mujoco.Renderer,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    cam: mujoco.MjvCamera,
    opt: mujoco.MjvOption,
) -> np.ndarray:
    """Render a single RGB frame."""
    renderer.update_scene(d, camera=cam, scene_option=opt)
    return renderer.render().copy()


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def run_phase(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    q_start: np.ndarray,
    q_end: np.ndarray,
    gripper_val: float,
    n_steps: int,
    renderer: mujoco.Renderer,
    cam: mujoco.MjvCamera,
    opt: mujoco.MjvOption,
    writer: imageio.core.format.Writer,
    dt_sim: float,
    fps: int,
    frames_per_step: int,
) -> None:
    """Linearly interpolate arm joint targets from current position to q_end.

    Uses d.qpos[:6] as the actual start (not q_start) so the trajectory is
    always continuous regardless of tracking lag from the previous phase.
    q_start is used only as a hint; actual start is from current qpos.
    """
    q_actual_start = d.qpos[:6].copy()
    for i in range(n_steps):
        alpha = i / max(n_steps - 1, 1)
        q_des = (1 - alpha) * q_actual_start + alpha * q_end

        # Gravity compensation feedforward
        grav_comp = d.qfrc_bias[:6] / KP_VEC
        d.ctrl[:6] = q_des + grav_comp
        d.ctrl[6] = gripper_val

        mujoco.mj_step(m, d)

        # Record every N sim steps to match target fps
        if i % frames_per_step == 0:
            frame = render_frame(renderer, m, d, cam, opt)
            writer.append_data(frame)


def hold_phase(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    q_hold: np.ndarray,
    gripper_val: float,
    n_steps: int,
    renderer: mujoco.Renderer,
    cam: mujoco.MjvCamera,
    opt: mujoco.MjvOption,
    writer: imageio.core.format.Writer,
    frames_per_step: int,
) -> None:
    """Hold a fixed configuration for n_steps simulation steps."""
    for i in range(n_steps):
        grav_comp = d.qfrc_bias[:6] / KP_VEC
        d.ctrl[:6] = q_hold + grav_comp
        d.ctrl[6] = gripper_val
        mujoco.mj_step(m, d)
        if i % frames_per_step == 0:
            frame = render_frame(renderer, m, d, cam, opt)
            writer.append_data(frame)


# ---------------------------------------------------------------------------
# Collision-free trajectory execution
# ---------------------------------------------------------------------------

def run_phase_planned(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    q_end: np.ndarray,
    gripper_val: float,
    steps_per_segment: int,
    renderer: mujoco.Renderer,
    cam: mujoco.MjvCamera,
    opt: mujoco.MjvOption,
    writer,
    frames_per_step: int,
    label: str = "",
) -> None:
    """Plan a collision-free path to q_end then execute it segment by segment.

    Uses Lab 4 RRT* + shortcutting to compute a sequence of collision-free
    waypoints, then runs `run_phase()` over each consecutive pair.  Each
    segment is guaranteed collision-free by the RRT* edge-check resolution
    (0.05 rad), so linear interpolation within a segment is safe.

    Args:
        m: MuJoCo model.
        d: MuJoCo data (current qpos[:6] used as start).
        q_end: Goal configuration (6,).
        gripper_val: Gripper control value (held constant).
        steps_per_segment: Sim steps allocated per waypoint segment.
            Automatically scaled by segment length.
        renderer, cam, opt, writer, frames_per_step: Rendering pipeline.
        label: Human-readable phase label for error messages.
    """
    q_start = d.qpos[:6].copy()
    waypoints = plan_collision_free(q_start, q_end, label=label)

    if len(waypoints) < 2:
        # Already at goal — nothing to do
        return

    # Distribute steps proportionally to joint-space distance per segment
    dists = [np.linalg.norm(waypoints[i + 1] - waypoints[i])
             for i in range(len(waypoints) - 1)]
    total_dist = sum(dists) or 1.0
    total_steps = steps_per_segment * len(dists)

    for i, (wp_start, wp_end) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        seg_steps = max(50, int(total_steps * dists[i] / total_dist))
        run_phase(m, d, wp_start, wp_end, gripper_val, seg_steps,
                  renderer, cam, opt, writer, None, None, frames_per_step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Lab 5 — Professional Pick-and-Place Demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build and compile scene
    # ------------------------------------------------------------------
    print("\n[1/6] Building MjSpec scene ...")
    spec = build_scene()
    m = spec.compile()
    d = mujoco.MjData(m)
    print(f"      Compiled: nq={m.nq}, nv={m.nv}, nu={m.nu}, nbody={m.nbody}")

    # Identify key IDs
    attach_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    pinch_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "2f85_pinch")
    box_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box_A")
    # Find box freejoint qpos start index
    box_jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
    box_qpos_start = m.jnt_qposadr[box_jnt_id]
    print(f"      attachment_site id={attach_id}, pinch id={pinch_id}")
    print(f"      box_A body_id={box_body_id}, box_qpos_start={box_qpos_start}")

    # ------------------------------------------------------------------
    # 2. Compute IK for key configurations
    # ------------------------------------------------------------------
    print("\n[2/6] Computing IK for key configurations ...")

    # Target rotation: attachment_site pointing straight down
    # At Q_HOME the end-effector faces -Y in world; we want it to face -Z (down)
    # Orientation: z-axis of site should point downward → R col2 = [0,0,-1]
    # Let's compute target rotation from Q_HOME position (site already faces down at HOME)
    d.qpos[:6] = Q_HOME
    mujoco.mj_forward(m, d)
    R_down = d.site_xmat[attach_id].reshape(3, 3).copy()
    print(f"      Rotation at Q_HOME:\n{R_down}")

    def ik_at(pos: np.ndarray, q_init: np.ndarray, label: str) -> np.ndarray:
        q = compute_ik(m, d, attach_id, pos, R_down, q_init)
        d.qpos[:6] = q
        mujoco.mj_forward(m, d)
        achieved = d.site_xpos[attach_id]
        err = np.linalg.norm(pos - achieved)
        print(f"      {label}: target={pos}, achieved={achieved}, err={err:.4f} m")
        return q

    # attachment_site must be at BOX_POS + [0,0, GRIPPER_TIP_OFFSET] to have pinch at box
    pregrasp_pos = BOX_A_POS + np.array([0, 0, GRIPPER_TIP_OFFSET + HOVER_HEIGHT])
    # Pinch at box centre (no dip): Robotiq pads land in the middle of the box sides
    grasp_pos    = BOX_A_POS + np.array([0, 0, GRIPPER_TIP_OFFSET])
    preplace_pos = BOX_B_POS + np.array([0, 0, GRIPPER_TIP_OFFSET + HOVER_HEIGHT])
    place_pos    = BOX_B_POS + np.array([0, 0, GRIPPER_TIP_OFFSET])

    q_pregrasp = ik_at(pregrasp_pos, Q_HOME,    "q_pregrasp")
    q_grasp    = ik_at(grasp_pos,    q_pregrasp, "q_grasp   ")
    # Seed preplace from Q_HOME: the Menagerie UR5e reaches Y=-0.20 via a small
    # shoulder_pan rotation from Q_HOME (-π/2 → less negative), not a mirrored
    # version of pregrasp (which produces a behind-the-arm config at +130° that
    # the collision checker rejects).
    q_preplace = ik_at(preplace_pos, Q_HOME,     "q_preplace")
    q_place    = ik_at(place_pos,    q_preplace,  "q_place   ")

    print(f"      q_pregrasp joints: {np.round(q_pregrasp, 3)}")
    print(f"      q_preplace joints: {np.round(q_preplace, 3)}")
    print(f"      CC q_preplace collision-free: {_cc.is_collision_free(q_preplace)}")
    print(f"      CC Q_HOME collision-free: {_cc.is_collision_free(Q_HOME)}")

    # ------------------------------------------------------------------
    # 3. Setup renderer + camera
    # ------------------------------------------------------------------
    print("\n[3/6] Setting up renderer ...")
    renderer = make_renderer(m, width=1280, height=720)
    setup_camera(renderer)

    cam = mujoco.MjvCamera()
    cam.lookat = np.array([0.35, 0.0, 0.45])
    cam.distance = 1.65
    cam.azimuth = 140.0
    cam.elevation = -22.0

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]  = False

    # ------------------------------------------------------------------
    # 4. Video writer setup
    # ------------------------------------------------------------------
    FPS = 60
    DT_SIM = m.opt.timestep
    SIM_STEPS_PER_FRAME = max(1, int(round(1.0 / (FPS * DT_SIM))))
    print(f"\n[4/6] Video writer: {OUTPUT_VIDEO}")
    print(f"      dt_sim={DT_SIM:.4f}, steps_per_frame={SIM_STEPS_PER_FRAME}")

    writer = imageio.get_writer(
        str(OUTPUT_VIDEO),
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_log_level="quiet",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )

    # ------------------------------------------------------------------
    # 5. Initialize simulation
    # ------------------------------------------------------------------
    print("\n[5/6] Running pick-and-place sequence ...")
    mujoco.mj_resetData(m, d)
    d.qpos[:6] = Q_HOME
    d.ctrl[:6] = Q_HOME.copy()
    d.ctrl[6] = 0.0
    mujoco.mj_forward(m, d)

    # Phase durations in sim steps (generous for smooth video)
    # At dt=0.002 s, 500 steps = 1 s real time
    STEPS_MOVE = 800       # movement phases — longer for smooth tracking
    STEPS_FAST = 600       # descent/lift transitions
    STEPS_HOLD = 500       # hold / gripper action — enough for PD to settle
    STEPS_SETTLE = 600     # hold + settle at grasp

    GRIPPER_OPEN   = 0.0
    GRIPPER_CLOSED = 200.0

    def log(label: str) -> None:
        ee = d.site_xpos[attach_id]
        print(f"      → {label:30s}  EE={np.round(ee, 4)}")

    # --- HOME (hold to settle) ---
    print("      → HOME hold ...")
    hold_phase(m, d, Q_HOME, GRIPPER_OPEN, STEPS_HOLD, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME)
    log("HOME settled")

    # --- HOME → PREGRASP (collision-free long move) ---
    print("      → HOME → PREGRASP ...")
    run_phase_planned(m, d, q_pregrasp, GRIPPER_OPEN, STEPS_MOVE, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME, "HOME→PREGRASP")
    hold_phase(m, d, q_pregrasp, GRIPPER_OPEN, STEPS_HOLD, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME)
    log("PREGRASP settled")

    # --- PREGRASP → GRASP (short vertical descent — direct interp is safe) ---
    print("      → PREGRASP → GRASP ...")
    run_phase(m, d, q_pregrasp, q_grasp, GRIPPER_OPEN, STEPS_FAST, renderer, cam, opt, writer, DT_SIM, FPS, SIM_STEPS_PER_FRAME)
    hold_phase(m, d, q_grasp, GRIPPER_OPEN, STEPS_HOLD, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME)
    log("GRASP settled")

    # --- CLOSE GRIPPER ---
    print("      → CLOSE GRIPPER ...")
    hold_phase(m, d, q_grasp, GRIPPER_CLOSED, STEPS_SETTLE, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME)
    log("GRIPPER CLOSED")

    # --- LIFT: GRASP → PREGRASP (short vertical lift — direct interp is safe) ---
    print("      → LIFT (GRASP → PREGRASP) ...")
    run_phase(m, d, q_grasp, q_pregrasp, GRIPPER_CLOSED, STEPS_FAST, renderer, cam, opt, writer, DT_SIM, FPS, SIM_STEPS_PER_FRAME)
    hold_phase(m, d, q_pregrasp, GRIPPER_CLOSED, STEPS_HOLD, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME)
    log("LIFTED to PREGRASP")

    # --- PREGRASP → HOME (collision-free long move) ---
    print("      → PREGRASP → HOME ...")
    run_phase_planned(m, d, Q_HOME, GRIPPER_CLOSED, STEPS_MOVE, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME, "PREGRASP→HOME")
    hold_phase(m, d, Q_HOME, GRIPPER_CLOSED, STEPS_HOLD // 2, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME)
    log("HOME (carrying box)")

    # --- HOME → PREPLACE (collision-free long move — shoulder sweeps ~π) ---
    print("      → HOME → PREPLACE ...")
    run_phase_planned(m, d, q_preplace, GRIPPER_CLOSED, STEPS_MOVE, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME, "HOME→PREPLACE")
    hold_phase(m, d, q_preplace, GRIPPER_CLOSED, STEPS_HOLD, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME)
    log("PREPLACE settled")

    # --- PREPLACE → PLACE (short vertical descent — direct interp is safe) ---
    print("      → PREPLACE → PLACE ...")
    run_phase(m, d, q_preplace, q_place, GRIPPER_CLOSED, STEPS_FAST, renderer, cam, opt, writer, DT_SIM, FPS, SIM_STEPS_PER_FRAME)
    hold_phase(m, d, q_place, GRIPPER_CLOSED, STEPS_HOLD, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME)
    log("PLACE settled")

    # --- OPEN GRIPPER (release) ---
    print("      → OPEN GRIPPER (release) ...")
    hold_phase(m, d, q_place, GRIPPER_OPEN, STEPS_SETTLE, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME)
    log("GRIPPER OPENED")

    # --- PLACE → PREPLACE (short vertical retract — direct interp is safe) ---
    print("      → PLACE → PREPLACE (retract) ...")
    run_phase(m, d, q_place, q_preplace, GRIPPER_OPEN, STEPS_FAST, renderer, cam, opt, writer, DT_SIM, FPS, SIM_STEPS_PER_FRAME)
    hold_phase(m, d, q_preplace, GRIPPER_OPEN, STEPS_HOLD, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME)
    log("PREPLACE settled (retracted)")

    # --- PREPLACE → HOME (collision-free long move) ---
    print("      → PREPLACE → HOME ...")
    run_phase_planned(m, d, Q_HOME, GRIPPER_OPEN, STEPS_MOVE, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME, "PREPLACE→HOME")
    hold_phase(m, d, Q_HOME, GRIPPER_OPEN, STEPS_HOLD, renderer, cam, opt, writer, SIM_STEPS_PER_FRAME)
    log("HOME (final)")

    writer.close()

    # ------------------------------------------------------------------
    # 6. Done
    # ------------------------------------------------------------------
    print(f"\n[6/6] Video saved to: {OUTPUT_VIDEO}")
    size_mb = OUTPUT_VIDEO.stat().st_size / 1e6
    print(f"      File size: {size_mb:.1f} MB")
    print("\nDone!")


if __name__ == "__main__":
    main()
