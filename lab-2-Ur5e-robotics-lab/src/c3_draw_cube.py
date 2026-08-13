"""C3: Draw a 3D Cube — Full UR5e Pipeline Demo.

Integrates all modules from the lab:
  A2  Forward Kinematics   Pinocchio FK for EE tracking
  A3  Jacobian             For IK and singularity awareness
  A4  Inverse Kinematics   DLS IK for Cartesian waypoints
  B1  Trajectory Gen       Quintic polynomial for smooth joint-space paths

Architecture
============
1. Path Planning     – 12 edges of a cube, each drawn as a quintic joint-space
                       trajectory between IK-solved waypoints.
2. IK                – Damped Least Squares IK for each cube vertex (8 vertices).
3. Position Control  – MuJoCo Menagerie UR5e has built-in PD position servos.
                       We augment ctrl with gravity compensation and velocity
                       feedforward for sub-mm tracking:
                         ctrl = q_des + g(q)/Kp + Kd*qd_des/Kp
4. MuJoCo Viewer     – launch_passive with real-time trail and cube wireframe overlay.

Run:
    python3 lab-2-Ur5e-robotics-lab/src/c3_draw_cube.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin

from ur5e_common import NUM_JOINTS, Q_HOME, load_pinocchio_model
from a4_inverse_kinematics import ik_damped_least_squares
from b1_trajectory_generation import quintic_trajectory, JointTrajectoryPoint

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Cube geometry — positioned in the UR5e reachable workspace, offset in Y
# to avoid arm-table collisions at all IK solutions.
CUBE_CENTER = np.array([0.3, 0.2, 0.55])    # centre of cube in world frame
CUBE_SIDE = 0.10                              # 10 cm side length
SEGMENT_DURATION = 2.0                        # seconds per edge

# Actuator gains (must match MuJoCo Menagerie UR5e model)
KP = np.array([2000, 2000, 2000, 500, 500, 500], dtype=float)
KD = np.array([400, 400, 400, 100, 100, 100], dtype=float)

# Visualisation
TRAIL_DECIMATE = 10
TRAIL_COLOR = np.array([1.0, 0.15, 0.15, 0.85], dtype=np.float32)
CUBE_COLOR = np.array([0.2, 0.80, 0.20, 0.50], dtype=np.float32)
EE_MARKER_COLOR = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Cube geometry
# ═══════════════════════════════════════════════════════════════════════════

def cube_vertices(center: np.ndarray, side: float) -> list[np.ndarray]:
    """Generate the 8 vertices of a cube.

    Vertex ordering:
        0-3: bottom face (z = center_z - side/2)
        4-7: top face    (z = center_z + side/2)

    Args:
        center: Centre of the cube (3,).
        side: Side length (m).

    Returns:
        List of 8 vertex positions (3,).
    """
    h = side / 2.0
    offsets = [
        [-h, -h, -h], [+h, -h, -h], [+h, +h, -h], [-h, +h, -h],  # bottom
        [-h, -h, +h], [+h, -h, +h], [+h, +h, +h], [-h, +h, +h],  # top
    ]
    return [center + np.array(o) for o in offsets]


def cube_edges() -> list[tuple[int, int]]:
    """Return the 12 edges of a cube as vertex index pairs.

    The order traces a path that visits all edges with minimal redundancy:
    bottom face, vertical pillars, top face.
    """
    return [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Vertical pillars
        (0, 4), (4, 5), (5, 1),
        # Back to top
        (5, 6), (6, 2),
        # Remaining top edges
        (6, 7), (7, 3),
        # Last pillar
        (7, 4),
    ]


def cube_drawing_path() -> list[int]:
    """Return a vertex visitation order that draws the entire cube continuously.

    This Eulerian-like path visits all 12 edges without lifting the pen.
    """
    edges = cube_edges()
    path = [edges[0][0]]
    for a, b in edges:
        path.append(b)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# IK solving for cube vertices
# ═══════════════════════════════════════════════════════════════════════════

def solve_cube_ik(vertices: list[np.ndarray],
                  path: list[int],
                  pin_model, pin_data, ee_fid: int) -> list[np.ndarray]:
    """Solve IK for each vertex in the drawing path.

    Args:
        vertices: Cube vertex positions.
        path: Vertex visitation order.
        pin_model, pin_data, ee_fid: Pinocchio model.

    Returns:
        List of joint configurations (one per path point).
    """
    # Use home EE orientation as target orientation
    pin.forwardKinematics(pin_model, pin_data, Q_HOME)
    pin.updateFramePlacements(pin_model, pin_data)
    R_target = pin_data.oMf[ee_fid].rotation.copy()

    solutions = []
    q_current = Q_HOME.copy()

    for i, vi in enumerate(path):
        p_target = vertices[vi]
        result = ik_damped_least_squares(
            pin_model, pin_data, ee_fid,
            p_target, R_target, q_current,
            max_iter=500, tol=1e-4,
        )
        if not result.success:
            print(f"    WARNING: IK failed for vertex {vi} "
                  f"(err={result.final_error:.6f})")
        solutions.append(result.q.copy())
        q_current = result.q.copy()

    return solutions


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory generation
# ═══════════════════════════════════════════════════════════════════════════

def build_cube_trajectory(joint_waypoints: list[np.ndarray],
                          dt: float = 0.002) -> list[JointTrajectoryPoint]:
    """Build a multi-segment quintic trajectory through IK waypoints.

    Args:
        joint_waypoints: Joint configurations at each vertex.
        dt: Simulation timestep.

    Returns:
        Complete trajectory as list of JointTrajectoryPoint.
    """
    all_points = []
    t_offset = 0.0

    for i in range(len(joint_waypoints) - 1):
        seg = quintic_trajectory(
            joint_waypoints[i], joint_waypoints[i + 1],
            SEGMENT_DURATION, dt,
        )
        for pt in seg:
            all_points.append(JointTrajectoryPoint(
                t=pt.t + t_offset, q=pt.q, qd=pt.qd, qdd=pt.qdd,
            ))
        t_offset += SEGMENT_DURATION

    return all_points


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _add_sphere(scn, pos, size: float, rgba):
    """Append a visual-only sphere to the scene."""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([size, 0.0, 0.0]),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    scn.ngeom += 1


def _add_line(scn, a, b, width: float, rgba):
    """Draw a thin capsule line between two 3D points."""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_connector(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
    )
    scn.geoms[scn.ngeom].rgba = rgba
    scn.ngeom += 1


def _draw_cube_wireframe(scn, vertices: list[np.ndarray]):
    """Draw the target cube as a green wireframe."""
    edges = cube_edges()
    for a, b in edges:
        _add_line(scn, vertices[a], vertices[b], 0.002, CUBE_COLOR)


def _draw_trail(scn, trail: list[np.ndarray]):
    """Render the accumulated EE trail (red spheres)."""
    for pos in trail:
        _add_sphere(scn, pos, 0.003, TRAIL_COLOR)


# ═══════════════════════════════════════════════════════════════════════════
# Main simulation loop
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("C3: Draw a 3D Cube — UR5e Full Pipeline Demo")
    print("=" * 72)

    # Load models
    pin_model, pin_data, ee_fid = load_pinocchio_model()

    # Generate cube geometry
    verts = cube_vertices(CUBE_CENTER, CUBE_SIDE)
    path = cube_drawing_path()
    n_edges = len(path) - 1

    print(f"\n  Cube centre: [{CUBE_CENTER[0]:.2f}, {CUBE_CENTER[1]:.2f}, {CUBE_CENTER[2]:.2f}]")
    print(f"  Side length: {CUBE_SIDE} m")
    print(f"  Vertices: {len(verts)}, Edges to draw: {n_edges}")
    print(f"  Total time: {n_edges * SEGMENT_DURATION:.1f} s")

    # Solve IK for cube vertices
    print("\n  Solving IK for cube path...")
    joint_wps = solve_cube_ik(verts, path, pin_model, pin_data, ee_fid)
    print(f"  Solved {len(joint_wps)} waypoints")

    # Build trajectory
    print("  Building trajectory...")
    dt = 0.002
    traj = build_cube_trajectory(joint_wps, dt)
    print(f"  {len(traj)} points, {traj[-1].t:.1f}s duration")

    # Set up MuJoCo simulation
    from ur5e_common import MJCF_SCENE_PATH
    model = mujoco.MjModel.from_xml_path(str(MJCF_SCENE_PATH))
    data = mujoco.MjData(model)

    # Initialise directly at the first trajectory waypoint
    data.qpos[:NUM_JOINTS] = traj[0].q
    data.qvel[:NUM_JOINTS] = 0.0
    data.ctrl[:NUM_JOINTS] = traj[0].q
    mujoco.mj_forward(model, data)

    # EE site ID
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    # Settle at the first waypoint for 3 s so the PD servo reaches
    # steady state before the trajectory begins.
    print("\n  Settling at start position (3 s)...")
    for _ in range(1500):
        g = data.qfrc_bias[:NUM_JOINTS].copy()
        data.ctrl[:NUM_JOINTS] = traj[0].q + g / KP
        mujoco.mj_step(model, data)

    # Reset simulation time so the trajectory starts at t = 0
    data.time = 0.0

    # Accumulators
    trail: list[np.ndarray] = []
    cart_errors: list[float] = []
    max_torque = 0.0
    edge_printed = 0

    # Launch viewer
    print("  Simulation running — close the viewer window to stop.\n")
    print(f"  {'Step':>7}  {'Time':>6}  {'|e_cart|':>10}  {'|tau|_max':>10}  Edge")
    print("  " + "-" * 55)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Camera setup — angled view to see the 3D cube
        viewer.cam.lookat[:] = CUBE_CENTER
        viewer.cam.distance = 1.2
        viewer.cam.elevation = -30.0
        viewer.cam.azimuth = 135.0

        step = 0

        while viewer.is_running() and step < len(traj):
            wall_t0 = time.time()

            tp = traj[step]

            # Position control with gravity compensation + velocity feedforward.
            # The Menagerie UR5e actuator applies:
            #   tau = Kp*(ctrl - qpos) - Kd*qvel
            # Setting ctrl = q_des + g/Kp + Kd*qd_des/Kp makes the effective
            # control law:
            #   tau ≈ Kp*(q_des - qpos) - Kd*(qvel - qd_des) + g(q)
            g = data.qfrc_bias[:NUM_JOINTS].copy()
            data.ctrl[:NUM_JOINTS] = tp.q + g / KP + KD * tp.qd / KP
            mujoco.mj_step(model, data)

            # Metrics
            ee = data.site_xpos[ee_site_id].copy()

            # Desired EE position from FK
            pin.forwardKinematics(pin_model, pin_data, tp.q)
            pin.updateFramePlacements(pin_model, pin_data)
            ee_des = pin_data.oMf[ee_fid].translation.copy()
            err = np.linalg.norm(ee_des - ee)

            cart_errors.append(err)
            act_force = data.qfrc_actuator[:NUM_JOINTS]
            max_torque = max(max_torque, float(np.max(np.abs(act_force))))

            # Trail
            if step % TRAIL_DECIMATE == 0:
                trail.append(ee.copy())

            # Render overlays
            viewer.user_scn.ngeom = 0
            _draw_cube_wireframe(viewer.user_scn, verts)
            _draw_trail(viewer.user_scn, trail)
            _add_sphere(viewer.user_scn, ee, 0.006, EE_MARKER_COLOR)

            viewer.sync()

            # Progress log (once per edge)
            current_edge = int(tp.t / SEGMENT_DURATION)
            if current_edge > edge_printed and current_edge <= n_edges:
                print(f"  {step:7d}  {tp.t:6.2f}  {err*1000:8.3f} mm  "
                      f"{max_torque:8.2f} Nm  edge {edge_printed+1}/{n_edges}")
                edge_printed = current_edge

            step += 1

            # Real-time pacing
            wall_elapsed = time.time() - wall_t0
            if wall_elapsed < dt:
                time.sleep(dt - wall_elapsed)

    # Summary
    if cart_errors:
        rms = math.sqrt(sum(e * e for e in cart_errors) / len(cart_errors))
        print(f"\n  {'=' * 55}")
        print("  C3: Draw Cube — Final Results")
        print(f"  {'=' * 55}")
        print(f"    RMS  Cartesian error  : {rms * 1000:.3f} mm")
        print(f"    Max  Cartesian error  : {max(cart_errors) * 1000:.3f} mm")
        print(f"    Final Cartesian error : {cart_errors[-1] * 1000:.3f} mm")
        print(f"    Max torque applied    : {max_torque:.2f} N·m")
        print(f"    Trail points rendered : {len(trail)}")
        print(f"    Trajectory steps      : {len(cart_errors)}")
        print(f"  {'=' * 55}")
    else:
        print("  Viewer closed before simulation started.")


if __name__ == "__main__":
    main()
