"""A5: Dynamics basics — reading qfrc_bias and qM from MuJoCo."""

from __future__ import annotations

import csv
import math
from pathlib import Path

MODEL_PATH = "models/two_link_torque.xml"
BIAS_CSV = Path("docs/a5_bias_cases.csv")
MASS_CSV = Path("docs/a5_mass_matrix_cases.csv")


def try_import_dependencies():
    """Import optional dependencies."""
    try:
        import mujoco
        import numpy as np
    except ModuleNotFoundError:
        return None, None
    return mujoco, np


def load_model_and_data(mujoco):
    """Load the torque model."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    return model, data


def set_state(model, data, qpos, qvel, gravity):
    """Set the same physical state with different gravity settings."""
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    model.opt.gravity[:] = gravity


def full_mass_matrix(mujoco, np, model, data):
    """Expand full inertia matrix from packed qM."""
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, mass, data.qM)
    return mass


def snapshot_dynamics(mujoco, np, gravity, qpos, qvel):
    """Take a dynamics snapshot for a given gravity/state."""
    model, data = load_model_and_data(mujoco)
    set_state(model, data, qpos, qvel, gravity)
    mujoco.mj_forward(model, data)

    return {
        "gravity": tuple(float(v) for v in gravity),
        "qpos_deg": tuple(math.degrees(float(v)) for v in data.qpos),
        "qvel_deg_s": tuple(math.degrees(float(v)) for v in data.qvel),
        "qfrc_bias": tuple(float(v) for v in data.qfrc_bias),
        "mass_matrix": full_mass_matrix(mujoco, np, model, data),
    }


def save_bias_cases_csv(snapshots):
    """Save gravity variation cases to CSV."""
    BIAS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with BIAS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "gravity_x", "gravity_y", "gravity_z", "bias_1", "bias_2"])
        for case_name, snapshot in snapshots:
            gravity = snapshot["gravity"]
            bias = snapshot["qfrc_bias"]
            writer.writerow(
                [
                    case_name,
                    f"{gravity[0]:.4f}",
                    f"{gravity[1]:.4f}",
                    f"{gravity[2]:.4f}",
                    f"{bias[0]:.10f}",
                    f"{bias[1]:.10f}",
                ]
            )


def save_mass_cases_csv(mass_cases):
    """Save mass matrix examples to CSV."""
    MASS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MASS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "m11", "m12", "m21", "m22"])
        for case_name, matrix in mass_cases:
            writer.writerow(
                [
                    case_name,
                    f"{float(matrix[0, 0]):.10f}",
                    f"{float(matrix[0, 1]):.10f}",
                    f"{float(matrix[1, 0]):.10f}",
                    f"{float(matrix[1, 1]):.10f}",
                ]
            )


def print_bias_report(snapshots):
    """Print qfrc_bias gravity effect report."""
    print("=" * 90)
    print("A5: Dynamics Basics")
    print("=" * 90)
    print("How does qfrc_bias change when gravity changes for the same q and qdot?")
    for case_name, snapshot in snapshots:
        print(
            f"- {case_name:14s} gravity={snapshot['gravity']} "
            f"-> qfrc_bias=[{snapshot['qfrc_bias'][0]:+.6f}, {snapshot['qfrc_bias'][1]:+.6f}]"
        )

    print("\nRemarks")
    print("1. `qfrc_bias` is the vector MuJoCo returns combining C(q,qdot)*qdot + g(q) terms.")
    print("2. Since this robot runs in the horizontal XY plane, adding z-gravity produces little or no torque around joint z-axes.")
    print("3. To clearly observe gravity effects on this model, apply gravity along an in-plane direction.")


def print_mass_report(np, mass_cases):
    """Show that the mass matrix is configuration-dependent."""
    print("\nMass matrix examples")
    for case_name, matrix in mass_cases:
        print(f"- {case_name}")
        print(f"  [{float(matrix[0, 0]):+.6f}  {float(matrix[0, 1]):+.6f}]")
        print(f"  [{float(matrix[1, 0]):+.6f}  {float(matrix[1, 1]):+.6f}]")

    diff = mass_cases[1][1] - mass_cases[0][1]
    frob = float(np.linalg.norm(diff))
    print(f"\n||M2 - M1||_F between configurations = {frob:.10f}")
    print("If non-zero, the inertia matrix is configuration-dependent.")


def conceptual_quiz_answer():
    """Brief conceptual answer."""
    print("\nConceptual quiz")
    print("Question: Why is M(q) configuration-dependent?")
    print("Answer: The masses and inertia axes of the links are distributed differently in space as the joint angles change.")
    print("This means the torque required to produce the same joint acceleration varies with the pose.")


def run_mujoco_dynamics_demo():
    """Generate dynamics report if MuJoCo is available."""
    mujoco, np = try_import_dependencies()
    if mujoco is None or np is None:
        print("MuJoCo and numpy cannot be imported in this environment. Script will be limited to conceptual notes.")
        conceptual_quiz_answer()
        return

    qpos = np.array([math.radians(40.0), math.radians(-30.0)])
    qvel = np.array([0.6, -0.4])

    snapshots = [
        ("gravity_off", snapshot_dynamics(mujoco, np, np.array([0.0, 0.0, 0.0]), qpos, qvel)),
        ("gravity_z", snapshot_dynamics(mujoco, np, np.array([0.0, 0.0, -9.81]), qpos, qvel)),
        ("gravity_y", snapshot_dynamics(mujoco, np, np.array([0.0, -9.81, 0.0]), qpos, qvel)),
    ]
    save_bias_cases_csv(snapshots)
    print_bias_report(snapshots)

    qvel_zero = np.array([0.0, 0.0])
    mass_cases = [
        (
            "q=[0deg, 0deg]",
            snapshot_dynamics(
                mujoco, np, np.array([0.0, 0.0, 0.0]), np.array([math.radians(0.0), math.radians(0.0)]), qvel_zero
            )["mass_matrix"],
        ),
        (
            "q=[90deg, -90deg]",
            snapshot_dynamics(
                mujoco, np, np.array([0.0, 0.0, 0.0]), np.array([math.radians(90.0), math.radians(-90.0)]), qvel_zero
            )["mass_matrix"],
        ),
    ]
    save_mass_cases_csv(mass_cases)
    print_mass_report(np, mass_cases)
    conceptual_quiz_answer()

    print(f"\nCSV outputs: {BIAS_CSV}, {MASS_CSV}")


if __name__ == "__main__":
    run_mujoco_dynamics_demo()
