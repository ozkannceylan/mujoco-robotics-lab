"""A5 — UR5e Dynamics: Mass matrix, gravity, Coriolis, RNEA, ABA.

Phase 2.3: Explores the dynamics equations of the UR5e using Pinocchio
for analytical computation and MuJoCo for cross-validation.

Covers:
    1. Mass matrix M(q) via pinocchio.crba()
    2. Gravity vector g(q) via pinocchio.computeGeneralizedGravity()
    3. Coriolis matrix C(q, v) via pinocchio.computeCoriolisMatrix()
    4. RNEA (inverse dynamics): torques to hold still at various configs
    5. ABA (forward dynamics): acceleration under gravity with zero torque
    6. Cross-validation: Pinocchio RNEA vs MuJoCo qfrc_bias
    7. Condition number of M(q) at different configurations
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ur5e_common import load_pinocchio_model, load_mujoco_model, Q_HOME, Q_ZEROS, NUM_JOINTS
from mujoco_sim import UR5eSimulator


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
TEST_CONFIGS: dict[str, np.ndarray] = {
    "zeros": Q_ZEROS.copy(),
    "home": Q_HOME.copy(),
    "elbow_up": np.array([0.0, -np.pi / 4, np.pi / 2, -np.pi / 4, -np.pi / 2, 0.0]),
    "shoulder_only": np.array([np.pi / 3, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "random": np.array([0.4, -1.1, 1.4, -1.3, 0.6, 0.5]),
}


def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# 1. Mass matrix M(q)
# ---------------------------------------------------------------------------

def compute_mass_matrix(
    model: pin.Model, data: pin.Data, q: np.ndarray
) -> np.ndarray:
    """Compute the joint-space mass (inertia) matrix M(q).

    Args:
        model: Pinocchio model.
        data: Pinocchio data.
        q: Joint configuration (NUM_JOINTS,).

    Returns:
        Symmetric positive-definite mass matrix (NUM_JOINTS, NUM_JOINTS).
    """
    pin.crba(model, data, q)
    # crba only fills the upper triangle; symmetrise
    M = data.M.copy()
    M = np.triu(M) + np.triu(M, k=1).T
    return M


def analyse_mass_matrix(M: np.ndarray, label: str) -> None:
    """Print properties of the mass matrix.

    Args:
        M: Mass matrix (n, n).
        label: Configuration name for display.
    """
    eigvals = np.linalg.eigvalsh(M)
    cond = np.linalg.cond(M)
    is_symmetric = np.allclose(M, M.T, atol=1e-10)
    is_pd = bool(np.all(eigvals > 0))

    print(f"\n--- M(q) at '{label}' ---")
    print(f"  Shape          : {M.shape}")
    print(f"  Symmetric      : {is_symmetric}")
    print(f"  Positive def.  : {is_pd}")
    print(f"  Eigenvalues    : [{', '.join(f'{v:.6f}' for v in eigvals)}]")
    print(f"  Condition num  : {cond:.2f}")
    print(f"  Diagonal (inertias): [{', '.join(f'{M[i,i]:.4f}' for i in range(M.shape[0]))}]")


# ---------------------------------------------------------------------------
# 2. Gravity vector g(q)
# ---------------------------------------------------------------------------

def compute_gravity_vector(
    model: pin.Model, data: pin.Data, q: np.ndarray
) -> np.ndarray:
    """Compute the generalized gravity vector g(q).

    Args:
        model: Pinocchio model.
        data: Pinocchio data.
        q: Joint configuration (NUM_JOINTS,).

    Returns:
        Gravity torques (NUM_JOINTS,).
    """
    pin.computeGeneralizedGravity(model, data, q)
    return data.g.copy()


# ---------------------------------------------------------------------------
# 3. Coriolis matrix C(q, v)
# ---------------------------------------------------------------------------

def compute_coriolis_matrix(
    model: pin.Model, data: pin.Data, q: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Compute the Coriolis matrix C(q, v).

    Args:
        model: Pinocchio model.
        data: Pinocchio data.
        q: Joint configuration (NUM_JOINTS,).
        v: Joint velocities (NUM_JOINTS,).

    Returns:
        Coriolis matrix (NUM_JOINTS, NUM_JOINTS).
    """
    pin.computeCoriolisMatrix(model, data, q, v)
    return data.C.copy()


# ---------------------------------------------------------------------------
# 4. RNEA — inverse dynamics
# ---------------------------------------------------------------------------

def compute_rnea(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Compute inverse dynamics via RNEA: tau = M*a + C*v + g.

    Args:
        model: Pinocchio model.
        data: Pinocchio data.
        q: Joint configuration (NUM_JOINTS,).
        v: Joint velocities (NUM_JOINTS,).
        a: Joint accelerations (NUM_JOINTS,).

    Returns:
        Required joint torques (NUM_JOINTS,).
    """
    return pin.rnea(model, data, q, v, a).copy()


# ---------------------------------------------------------------------------
# 5. ABA — forward dynamics
# ---------------------------------------------------------------------------

def compute_aba(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    """Compute forward dynamics via ABA: qdd = M^-1 * (tau - C*v - g).

    Args:
        model: Pinocchio model.
        data: Pinocchio data.
        q: Joint configuration (NUM_JOINTS,).
        v: Joint velocities (NUM_JOINTS,).
        tau: Applied joint torques (NUM_JOINTS,).

    Returns:
        Resulting joint accelerations (NUM_JOINTS,).
    """
    return pin.aba(model, data, q, v, tau).copy()


# ---------------------------------------------------------------------------
# 6. Cross-validation: Pinocchio RNEA vs MuJoCo qfrc_bias
# ---------------------------------------------------------------------------

def cross_validate_gravity(
    model: pin.Model,
    data: pin.Data,
    sim: UR5eSimulator,
    q: np.ndarray,
    label: str,
) -> float:
    """Compare Pinocchio RNEA (v=0, a=0) with MuJoCo qfrc_bias.

    When v=0 and a=0, RNEA returns g(q), and MuJoCo qfrc_bias = C*v + g = g.

    Args:
        model: Pinocchio model.
        data: Pinocchio data.
        sim: MuJoCo simulator wrapper.
        q: Joint configuration (NUM_JOINTS,).
        label: Configuration name for display.

    Returns:
        Maximum absolute error between the two.
    """
    v_zero = np.zeros(NUM_JOINTS)
    a_zero = np.zeros(NUM_JOINTS)

    # Pinocchio: RNEA at (q, v=0, a=0) gives gravity torques
    tau_pin = compute_rnea(model, data, q, v_zero, a_zero)

    # MuJoCo: qfrc_bias at v=0 gives gravity torques
    sim.set_qpos(q)
    sim.set_qvel(np.zeros(NUM_JOINTS))
    mj_bias = sim.get_qfrc_bias()

    error = np.abs(tau_pin - mj_bias)
    max_err = float(np.max(error))

    print(f"\n--- Cross-validation at '{label}' ---")
    print(f"  Pinocchio RNEA  : [{', '.join(f'{v:+.4f}' for v in tau_pin)}]")
    print(f"  MuJoCo qfrc_bias: [{', '.join(f'{v:+.4f}' for v in mj_bias)}]")
    print(f"  Abs error       : [{', '.join(f'{v:.6f}' for v in error)}]")
    print(f"  Max error       : {max_err:.6f} Nm")
    print(f"  Match (atol=0.1): {np.allclose(tau_pin, mj_bias, atol=0.1)}")

    return max_err


# ---------------------------------------------------------------------------
# 7. Condition number sweep
# ---------------------------------------------------------------------------

def condition_number_sweep(
    model: pin.Model, data: pin.Data, configs: dict[str, np.ndarray]
) -> None:
    """Print M(q) condition number at each configuration.

    Args:
        model: Pinocchio model.
        data: Pinocchio data.
        configs: Dictionary mapping config names to joint angles.
    """
    print_header("Condition Number Sweep")
    print(f"  {'Config':<20s} {'Cond(M)':<15s} {'Min eig':<12s} {'Max eig':<12s}")
    print("  " + "-" * 59)

    for name, q in configs.items():
        M = compute_mass_matrix(model, data, q)
        eigvals = np.linalg.eigvalsh(M)
        cond = np.linalg.cond(M)
        print(f"  {name:<20s} {cond:<15.2f} {eigvals[0]:<12.6f} {eigvals[-1]:<12.6f}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all dynamics analyses and print results."""
    # Load models
    print("Loading Pinocchio model...")
    model, data, _ = load_pinocchio_model()
    print(f"  nq = {model.nq}, nv = {model.nv}")

    print("Loading MuJoCo simulator...")
    sim = UR5eSimulator()
    print("  Ready.")

    # ------------------------------------------------------------------
    # 1. Mass matrix analysis
    # ------------------------------------------------------------------
    print_header("1. Mass Matrix M(q)")

    for name, q in TEST_CONFIGS.items():
        M = compute_mass_matrix(model, data, q)
        analyse_mass_matrix(M, name)

    # ------------------------------------------------------------------
    # 2. Gravity vector g(q)
    # ------------------------------------------------------------------
    print_header("2. Gravity Vector g(q)")

    for name, q in TEST_CONFIGS.items():
        g = compute_gravity_vector(model, data, q)
        print(f"\n  g(q) at '{name}':")
        print(f"    [{', '.join(f'{v:+.4f}' for v in g)}]")
        print(f"    Max |g_i| = {np.max(np.abs(g)):.4f} Nm (joint {np.argmax(np.abs(g)) + 1})")

    # ------------------------------------------------------------------
    # 3. Coriolis matrix C(q, v)
    # ------------------------------------------------------------------
    print_header("3. Coriolis Matrix C(q, v)")

    # Use a nonzero velocity to see nontrivial Coriolis effects
    v_test = np.array([0.2, -0.3, 0.5, -0.2, 0.1, 0.0])

    for name, q in TEST_CONFIGS.items():
        C = compute_coriolis_matrix(model, data, q, v_test)
        c_tau = C @ v_test  # Coriolis torque contribution
        print(f"\n  C(q, v) @ v at '{name}':")
        print(f"    [{', '.join(f'{v:+.6f}' for v in c_tau)}]")
        print(f"    Frobenius norm of C: {np.linalg.norm(C, 'fro'):.6f}")

    # Verify skew-symmetry of N = M_dot - 2*C
    print("\n  Verifying skew-symmetry of (M_dot - 2C):")
    q_test = TEST_CONFIGS["random"]
    eps = 1e-7
    M1 = compute_mass_matrix(model, data, q_test)
    M2 = compute_mass_matrix(model, data, q_test + eps * v_test)
    M_dot_approx = (M2 - M1) / eps
    C = compute_coriolis_matrix(model, data, q_test, v_test)
    N = M_dot_approx - 2.0 * C
    skew_err = np.linalg.norm(N + N.T, 'fro')
    print(f"    ||N + N^T||_F = {skew_err:.8f} (should be ~0)")

    # ------------------------------------------------------------------
    # 4. RNEA: torques to hold still
    # ------------------------------------------------------------------
    print_header("4. RNEA — Torques to Hold Still (tau = g(q))")

    v_zero = np.zeros(NUM_JOINTS)
    a_zero = np.zeros(NUM_JOINTS)

    for name, q in TEST_CONFIGS.items():
        tau_hold = compute_rnea(model, data, q, v_zero, a_zero)
        g = compute_gravity_vector(model, data, q)
        print(f"\n  '{name}':")
        print(f"    RNEA(q, 0, 0) : [{', '.join(f'{v:+.4f}' for v in tau_hold)}]")
        print(f"    g(q)           : [{', '.join(f'{v:+.4f}' for v in g)}]")
        print(f"    Match          : {np.allclose(tau_hold, g, atol=1e-10)}")

    # ------------------------------------------------------------------
    # 5. ABA: forward dynamics (acceleration under gravity)
    # ------------------------------------------------------------------
    print_header("5. ABA — Forward Dynamics (Zero Torque, Gravity Only)")

    tau_zero = np.zeros(NUM_JOINTS)

    for name, q in TEST_CONFIGS.items():
        qdd = compute_aba(model, data, q, v_zero, tau_zero)
        print(f"\n  '{name}':")
        print(f"    qdd = [{', '.join(f'{v:+.4f}' for v in qdd)}] rad/s^2")
        print(f"    Max |qdd_i| = {np.max(np.abs(qdd)):.4f} rad/s^2 (joint {np.argmax(np.abs(qdd)) + 1})")

    # Verify ABA = M^-1 * (tau - g) when v=0
    print("\n  Verifying ABA = M^-1 * (tau - g):")
    q_check = TEST_CONFIGS["home"]
    M = compute_mass_matrix(model, data, q_check)
    g = compute_gravity_vector(model, data, q_check)
    qdd_aba = compute_aba(model, data, q_check, v_zero, tau_zero)
    qdd_manual = np.linalg.solve(M, tau_zero - g)
    print(f"    ABA    : [{', '.join(f'{v:+.6f}' for v in qdd_aba)}]")
    print(f"    Manual : [{', '.join(f'{v:+.6f}' for v in qdd_manual)}]")
    print(f"    Match  : {np.allclose(qdd_aba, qdd_manual, atol=1e-8)}")

    # ------------------------------------------------------------------
    # 6. Cross-validation: Pinocchio vs MuJoCo
    # ------------------------------------------------------------------
    print_header("6. Cross-Validation: Pinocchio RNEA vs MuJoCo qfrc_bias")

    max_errors = []
    for name, q in TEST_CONFIGS.items():
        err = cross_validate_gravity(model, data, sim, q, name)
        max_errors.append(err)

    overall_max = max(max_errors)
    print(f"\n  Overall max error across configs: {overall_max:.6f} Nm")
    if overall_max < 0.1:
        print("  [PASS] Pinocchio and MuJoCo dynamics agree within tolerance.")
    else:
        print("  [WARN] Significant mismatch detected — check model parameters.")

    # ------------------------------------------------------------------
    # 7. Condition number sweep
    # ------------------------------------------------------------------
    condition_number_sweep(model, data, TEST_CONFIGS)

    print("\n" + "=" * 70)
    print("  A5 Dynamics analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
