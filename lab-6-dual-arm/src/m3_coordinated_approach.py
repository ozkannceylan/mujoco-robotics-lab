"""M3 — Coordinated approach: both arms move to grasp standoff poses.

Two-phase motion using joint PD control (M1 controller):
  Phase 1: HOME → approach poses (10 cm standoff from box surface)
  Phase 2: approach → grasp standoff poses (5 cm standoff)

IK solutions are searched for collision-free configurations. Grasp IK
is seeded from the approach IK so the Phase 2 transition is small.

Gate criteria:
  - Both IK solutions converge
  - Final EE position error < 5 mm per arm
  - Arrival time difference < 50 ms between arms
  - EE z-axis dot with toward-box direction > 0.9
  - Video: media/m3_approach.mp4
  - Screenshot: media/m3_final.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import imageio
import mujoco
import numpy as np
import pinocchio as pin

from dual_arm_model import DualArmModel
from grasp_pose_calculator import compute_grasp_poses
from joint_pd_controller import DualArmJointPD
from lab6_common import (
    DT,
    LEFT_JOINT_SLICE,
    MEDIA_DIR,
    NUM_JOINTS_PER_ARM,
    Q_HOME_LEFT,
    Q_HOME_RIGHT,
    RIGHT_JOINT_SLICE,
    SCENE_DUAL_PATH,
    get_mj_site_id,
)

# Video settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
FRAME_SKIP = int(1.0 / (DT * VIDEO_FPS))

# Controller gains — higher than M1 to handle large reconfigurations
KP = 500.0
KD = 50.0

# Timing
PHASE_MAX_TIME = 8.0
SETTLE_THRESHOLD = 0.005  # rad
SETTLE_DURATION = 0.5
INITIAL_SETTLE = 1.0


def _wrap_joints(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Wrap each joint to the equivalent angle closest to q_ref."""
    q_out = q.copy()
    for i in range(len(q)):
        diff = q[i] - q_ref[i]
        q_out[i] = q_ref[i] + ((diff + np.pi) % (2 * np.pi) - np.pi)
    return q_out


def _find_collision_free_ik(
    dual: DualArmModel,
    arm: str,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    q_ref: np.ndarray,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    q_init: np.ndarray | None = None,
    n_trials: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray | None, bool]:
    """Search for a collision-free IK solution closest to q_init.

    Args:
        dual: DualArmModel instance.
        arm: "left" or "right".
        target_pos: Target EE position in world frame.
        target_rot: Target EE rotation matrix.
        q_ref: Reference for joint wrapping (usually Q_HOME).
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        q_init: Center of random search / distance metric. Defaults to q_ref.
        n_trials: Number of random seeds to try.
        seed: RNG seed.

    Returns:
        Tuple of (best_q, found).
    """
    rng = np.random.default_rng(seed)
    model = dual.model_left if arm == "left" else dual.model_right
    data = dual.data_left if arm == "left" else dual.data_right
    ee_fid = dual.ee_frame_id_left if arm == "left" else dual.ee_frame_id_right
    base = dual.base_left if arm == "left" else dual.base_right
    jslice = LEFT_JOINT_SLICE if arm == "left" else RIGHT_JOINT_SLICE
    center = q_init if q_init is not None else q_ref

    best_q: np.ndarray | None = None
    best_dist = float("inf")

    for trial in range(n_trials):
        if trial == 0:
            qi = center.copy()
        else:
            qi = center + rng.uniform(-2.5, 2.5, size=NUM_JOINTS_PER_ARM)

        q_sol, ok, _ = dual._ik_single(
            model, data, ee_fid,
            target_pos - base, target_rot, qi,
            0.01, 200, 1e-4, 1e-3, 0.5,
        )
        if not ok:
            continue

        q_w = _wrap_joints(q_sol, q_ref)
        # Verify FK still matches after wrapping
        fk = dual.fk(arm, q_w)
        if np.linalg.norm(fk.translation - target_pos) > 0.001:
            continue

        # Check collision at this configuration
        saved = mj_data.qpos[jslice].copy()
        mj_data.qpos[jslice] = q_w
        mujoco.mj_forward(mj_model, mj_data)

        bad = 0
        for c in range(mj_data.ncon):
            ct = mj_data.contact[c]
            if np.isnan(ct.dist) or ct.dist > -0.001:
                continue
            b1 = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                    mj_model.geom_bodyid[ct.geom1]) or ""
            b2 = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                    mj_model.geom_bodyid[ct.geom2]) or ""
            # box-table contact is normal
            if ("table" in b1 and "box" in b2) or ("box" in b1 and "table" in b2):
                continue
            if arm in b1 or arm in b2:
                bad += 1

        mj_data.qpos[jslice] = saved
        mujoco.mj_forward(mj_model, mj_data)

        if bad > 0:
            continue

        dist = np.linalg.norm(q_w - center)
        if dist < best_dist:
            best_dist = dist
            best_q = q_w.copy()

    return best_q, best_q is not None


def run_phase(
    label: str,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    controller: DualArmJointPD,
    renderer: mujoco.Renderer,
    q_target_left: np.ndarray,
    q_target_right: np.ndarray,
    frames: list[np.ndarray],
) -> dict:
    """Run a PD phase until both arms settle or timeout.

    Returns:
        Dict with settle times and final errors.
    """
    n_max = int(PHASE_MAX_TIME / DT)
    n_settle_req = int(SETTLE_DURATION / DT)

    left_settled_count = 0
    right_settled_count = 0
    left_settle_time = None
    right_settle_time = None

    for step in range(n_max):
        controller.compute(mj_data, q_target_left, q_target_right)
        mujoco.mj_step(mj_model, mj_data)

        t = (step + 1) * DT

        err_left = np.max(np.abs(q_target_left - mj_data.qpos[LEFT_JOINT_SLICE]))
        err_right = np.max(np.abs(q_target_right - mj_data.qpos[RIGHT_JOINT_SLICE]))

        if err_left < SETTLE_THRESHOLD:
            left_settled_count += 1
        else:
            left_settled_count = 0
        if err_right < SETTLE_THRESHOLD:
            right_settled_count += 1
        else:
            right_settled_count = 0

        if left_settle_time is None and left_settled_count >= n_settle_req:
            left_settle_time = t - SETTLE_DURATION
        if right_settle_time is None and right_settled_count >= n_settle_req:
            right_settle_time = t - SETTLE_DURATION

        if step % FRAME_SKIP == 0:
            renderer.update_scene(mj_data)
            frames.append(renderer.render().copy())

        # Both settled — run 0.5s more for video then break
        if left_settle_time is not None and right_settle_time is not None:
            extra_steps = int(0.5 / DT)
            for s2 in range(extra_steps):
                controller.compute(mj_data, q_target_left, q_target_right)
                mujoco.mj_step(mj_model, mj_data)
                if s2 % FRAME_SKIP == 0:
                    renderer.update_scene(mj_data)
                    frames.append(renderer.render().copy())
            break

    return {
        "left_settle_time": left_settle_time,
        "right_settle_time": right_settle_time,
        "left_final_err": np.max(np.abs(q_target_left - mj_data.qpos[LEFT_JOINT_SLICE])),
        "right_final_err": np.max(np.abs(q_target_right - mj_data.qpos[RIGHT_JOINT_SLICE])),
    }


def main() -> None:
    """Run M3 coordinated approach."""
    print("=" * 70)
    print("M3: Coordinated Approach to Box")
    print("=" * 70)

    # Load models
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE_DUAL_PATH))
    mj_data = mujoco.MjData(mj_model)
    dual = DualArmModel()
    controller = DualArmJointPD(kp=KP, kd=KD)

    # Initialize at home
    mj_data.qpos[LEFT_JOINT_SLICE] = Q_HOME_LEFT
    mj_data.qpos[RIGHT_JOINT_SLICE] = Q_HOME_RIGHT
    mujoco.mj_forward(mj_model, mj_data)

    # ---- Compute poses from box state ----
    poses = compute_grasp_poses(mj_model, mj_data)
    box_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_center = mj_data.xpos[box_id].copy()

    print("\n  Target poses (from box):")
    for name, pose in poses.items():
        print(f"    {name:20s}: pos={pose.translation.round(3)}")

    # ---- Solve IK: collision-free, chained approach→grasp ----
    print("\n  Solving collision-free IK (approach, then grasp seeded from approach)...")

    # Approach IK — closest to HOME, collision-free
    q_left_approach, ok_la = _find_collision_free_ik(
        dual, "left", poses["left_approach"].translation, poses["left_approach"].rotation,
        Q_HOME_LEFT, mj_model, mj_data, seed=12345,
    )
    q_right_approach, ok_ra = _find_collision_free_ik(
        dual, "right", poses["right_approach"].translation, poses["right_approach"].rotation,
        Q_HOME_RIGHT, mj_model, mj_data, seed=12345,
    )

    if not ok_la or not ok_ra:
        print(f"    *** Approach IK FAILED: left={ok_la}, right={ok_ra} ***")
        return

    print(f"    left_approach:  q={q_left_approach.round(3)}")
    print(f"    right_approach: q={q_right_approach.round(3)}")

    # Grasp IK — seeded from approach IK, collision-free
    q_left_grasp, ok_lg = _find_collision_free_ik(
        dual, "left", poses["left_grasp"].translation, poses["left_grasp"].rotation,
        Q_HOME_LEFT, mj_model, mj_data, q_init=q_left_approach, seed=12345,
    )
    q_right_grasp, ok_rg = _find_collision_free_ik(
        dual, "right", poses["right_grasp"].translation, poses["right_grasp"].rotation,
        Q_HOME_RIGHT, mj_model, mj_data, q_init=q_right_approach, seed=12345,
    )

    if not ok_lg or not ok_rg:
        print(f"    *** Grasp IK FAILED: left={ok_lg}, right={ok_rg} ***")
        return

    print(f"    left_grasp:     q={q_left_grasp.round(3)}")
    print(f"    right_grasp:    q={q_right_grasp.round(3)}")

    all_ik_ok = True

    # Print approach→grasp distance
    d_l = np.linalg.norm(q_left_grasp - q_left_approach)
    d_r = np.linalg.norm(q_right_grasp - q_right_approach)
    print(f"\n    Approach→grasp joint distance: L={d_l:.3f} rad, R={d_r:.3f} rad")

    # ---- Simulation ----
    renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    frames: list[np.ndarray] = []

    wall_start = time.time()

    # Settle at home
    print(f"\n  Settling at home for {INITIAL_SETTLE}s...")
    n_settle = int(INITIAL_SETTLE / DT)
    for step in range(n_settle):
        controller.compute(mj_data, Q_HOME_LEFT, Q_HOME_RIGHT)
        mujoco.mj_step(mj_model, mj_data)
        if step % FRAME_SKIP == 0:
            renderer.update_scene(mj_data)
            frames.append(renderer.render().copy())

    # Phase 1: HOME → approach
    print("\n  Phase 1: HOME → approach poses")
    p1 = run_phase(
        "Phase 1", mj_model, mj_data, controller, renderer,
        q_left_approach, q_right_approach, frames,
    )
    p1_arrival_diff = None
    if p1["left_settle_time"] is not None and p1["right_settle_time"] is not None:
        p1_arrival_diff = abs(p1["left_settle_time"] - p1["right_settle_time"])
        print(f"    Left settled:  {p1['left_settle_time']:.3f}s")
        print(f"    Right settled: {p1['right_settle_time']:.3f}s")
        print(f"    Arrival diff:  {p1_arrival_diff*1000:.1f} ms")
    else:
        print(f"    WARNING: Arm(s) did not settle within {PHASE_MAX_TIME}s")
        print(f"    Left:  {'settled' if p1['left_settle_time'] else 'NOT settled'}")
        print(f"    Right: {'settled' if p1['right_settle_time'] else 'NOT settled'}")

    # Phase 2: approach → grasp standoff
    print("\n  Phase 2: approach → grasp standoff poses")
    p2 = run_phase(
        "Phase 2", mj_model, mj_data, controller, renderer,
        q_left_grasp, q_right_grasp, frames,
    )
    p2_arrival_diff = None
    if p2["left_settle_time"] is not None and p2["right_settle_time"] is not None:
        p2_arrival_diff = abs(p2["left_settle_time"] - p2["right_settle_time"])
        print(f"    Left settled:  {p2['left_settle_time']:.3f}s")
        print(f"    Right settled: {p2['right_settle_time']:.3f}s")
        print(f"    Arrival diff:  {p2_arrival_diff*1000:.1f} ms")
    else:
        print(f"    WARNING: Arm(s) did not settle within {PHASE_MAX_TIME}s")

    renderer.close()
    wall_elapsed = time.time() - wall_start
    print(f"\n  Simulation: {len(frames)} frames, {wall_elapsed:.1f}s wall time")

    # ---- Final Cartesian error check ----
    left_ee_sid = get_mj_site_id(mj_model, "left_ee_site")
    right_ee_sid = get_mj_site_id(mj_model, "right_ee_site")

    mj_left_pos = mj_data.site_xpos[left_ee_sid].copy()
    mj_right_pos = mj_data.site_xpos[right_ee_sid].copy()
    target_left_pos = poses["left_grasp"].translation
    target_right_pos = poses["right_grasp"].translation

    cart_err_left = np.linalg.norm(mj_left_pos - target_left_pos)
    cart_err_right = np.linalg.norm(mj_right_pos - target_right_pos)

    # Orientation check: EE z-axis dot with direction toward box
    R_left = mj_data.site_xmat[left_ee_sid].reshape(3, 3)
    R_right = mj_data.site_xmat[right_ee_sid].reshape(3, 3)
    z_left = R_left[:, 2]
    z_right = R_right[:, 2]

    dir_left = box_center - mj_left_pos
    dir_left /= np.linalg.norm(dir_left)
    dir_right = box_center - mj_right_pos
    dir_right /= np.linalg.norm(dir_right)

    dot_left = np.dot(z_left, dir_left)
    dot_right = np.dot(z_right, dir_right)

    print(f"\n  Final Cartesian state:")
    print(f"    Left  EE pos:  {mj_left_pos.round(4)}  target: {target_left_pos.round(4)}"
          f"  err: {cart_err_left*1000:.2f} mm")
    print(f"    Right EE pos:  {mj_right_pos.round(4)}  target: {target_right_pos.round(4)}"
          f"  err: {cart_err_right*1000:.2f} mm")
    print(f"    Left  EE z-axis dot toward box: {dot_left:.4f}")
    print(f"    Right EE z-axis dot toward box: {dot_right:.4f}")

    # ---- Save video ----
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    video_path = MEDIA_DIR / "m3_approach.mp4"
    print(f"\n  Writing video to {video_path} ...")
    writer = imageio.get_writer(
        str(video_path), fps=VIDEO_FPS, codec="libx264",
        quality=8, macro_block_size=1,
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"  Video saved: {video_path}")

    # ---- Save final screenshot ----
    screenshot_renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    screenshot_renderer.update_scene(mj_data)
    img = screenshot_renderer.render()
    screenshot_path = MEDIA_DIR / "m3_final.png"
    imageio.imwrite(str(screenshot_path), img)
    screenshot_renderer.close()
    print(f"  Screenshot saved: {screenshot_path}")

    # ---- Gate ----
    print("\n" + "=" * 70)
    print("M3 GATE CRITERIA")
    print("=" * 70)

    g1 = all_ik_ok
    g2 = cart_err_left * 1000 < 5.0 and cart_err_right * 1000 < 5.0
    g3_p1 = p1_arrival_diff is not None and p1_arrival_diff * 1000 < 50
    g3_p2 = p2_arrival_diff is not None and p2_arrival_diff * 1000 < 50
    g3 = g3_p1 and g3_p2
    g4 = dot_left > 0.9 and dot_right > 0.9
    g5 = video_path.exists()
    g6 = screenshot_path.exists()

    print(f"\n  [1] All IK converge:            {'PASS' if g1 else 'FAIL'}")
    print(f"  [2] Cartesian error < 5mm:      {'PASS' if g2 else 'FAIL'}"
          f"  (L={cart_err_left*1000:.2f}mm, R={cart_err_right*1000:.2f}mm)")
    print(f"  [3] Arrival diff < 50ms:")
    if p1_arrival_diff is not None:
        print(f"       Phase 1: {'PASS' if g3_p1 else 'FAIL'}"
              f"  ({p1_arrival_diff*1000:.1f} ms)")
    else:
        print(f"       Phase 1: FAIL  (arm(s) did not settle)")
    if p2_arrival_diff is not None:
        print(f"       Phase 2: {'PASS' if g3_p2 else 'FAIL'}"
              f"  ({p2_arrival_diff*1000:.1f} ms)")
    else:
        print(f"       Phase 2: FAIL  (arm(s) did not settle)")
    print(f"  [4] EE z-dot toward box > 0.9:  {'PASS' if g4 else 'FAIL'}"
          f"  (L={dot_left:.4f}, R={dot_right:.4f})")
    print(f"  [5] Video saved:                {'PASS' if g5 else 'FAIL'}")
    print(f"  [6] Screenshot saved:           {'PASS' if g6 else 'FAIL'}")

    gate_pass = g1 and g2 and g3 and g4 and g5 and g6
    print(f"\n  M3 GATE: {'PASSED' if gate_pass else 'FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
