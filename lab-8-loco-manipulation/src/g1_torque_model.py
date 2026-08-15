"""Lab 8 — Torque-actuated Unitree G1 model (M0 Step 0.1).

The Menagerie `g1.xml` ships 29 **position** servos::

    <position class="g1" kp="500" dampratio="1" inheritrange="1" .../>

Lab 7 proved those servos cannot track a dynamic walking reference (M4 blocked,
M3e failed 6 times: IK converges, PD replay diverges). Lab 8's whole-body QP →
RNEA pipeline outputs *joint torques*, so the first thing this lab needs is
command authority over torque.

This module builds that variant programmatically with `mujoco.MjSpec`, rather
than committing a hand-edited copy of `g1.xml`:

* Menagerie stays the single source of truth for kinematics, inertias and
  meshes — an upstream update flows through instead of silently diverging from
  a stale fork (Lab 5's L-6.1c: the analytical model must model the simulated
  body, and two copies of one robot is how that goes wrong).
* No mesh-path breakage. `g1.xml` declares `meshdir="assets"`, which MuJoCo
  resolves relative to the *top-level* model file; a copy under this lab's
  `models/` would need a symlink shim (see Lab 2's `models/assets`).
* It matches the repo convention already used by Labs 3–4
  (`build_mujoco_scene_spec`).

`export_xml()` can still write a compiled snapshot for inspection, but the
runtime path never depends on it.

Conversion details
------------------
`MjsActuator.set_to_motor()` switches gaintype/biastype to fixed/none with
gainprm[0] = 1, i.e. ``force = ctrl``. Each actuator's ``ctrlrange`` is then
set from its joint's ``actuatorfrcrange`` (the Unitree spec limits: 88 N·m
hips/yaw, 139 N·m hip-roll/knee, 50 N·m ankles, 25–88 N·m upper body), and
``ctrllimited`` is enabled so a runaway controller saturates instead of
teleporting the robot.

The Menagerie "stand" keyframe carries a ``ctrl`` vector of *position* targets
(0.2, 1.28 …). Under torque actuators those numbers would be interpreted as
newton-metres, so the keyframe's ctrl is zeroed while its qpos is kept.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

__all__ = [
    "build_g1_torque_spec",
    "compile_g1_torque_model",
    "export_xml",
    "torque_limits",
]


def _joint_force_ranges(spec: mujoco.MjSpec) -> dict[str, np.ndarray]:
    """Map joint name → actuator force range declared in the MJCF."""
    ranges: dict[str, np.ndarray] = {}
    for joint in spec.joints:
        if not joint.name:
            continue
        frc = np.asarray(joint.actfrcrange, dtype=float)
        if frc.shape == (2,) and not np.allclose(frc, 0.0):
            ranges[joint.name] = frc
    return ranges


def build_g1_torque_spec(
    g1_mjcf_path: Path,
    with_floor: bool = True,
    timestep: float | None = None,
) -> mujoco.MjSpec:
    """Load Menagerie G1 and convert its position servos to torque motors.

    Args:
        g1_mjcf_path: Path to Menagerie `g1.xml`.
        with_floor: Add a ground plane + light (g1.xml alone has no floor;
            upstream puts those in `scene.xml`, which we do not use because it
            re-includes the position-actuated robot).
        timestep: Optional simulation timestep override [s].

    Returns:
        An uncompiled `MjSpec` whose 29 actuators are torque motors.

    Raises:
        FileNotFoundError: If the Menagerie model is missing (run
            `tools/setup_env.sh`).
        RuntimeError: If an actuator has no resolvable force range.
    """
    if not Path(g1_mjcf_path).exists():
        raise FileNotFoundError(
            f"Menagerie G1 not found at {g1_mjcf_path}.\n"
            "Populate it with:  ./tools/setup_env.sh"
        )

    spec = mujoco.MjSpec.from_file(str(g1_mjcf_path))
    spec.modelname = "g1_29dof_torque"

    force_ranges = _joint_force_ranges(spec)

    for actuator in spec.actuators:
        # Menagerie names each actuator after the joint it drives.
        frc = force_ranges.get(actuator.name)
        if frc is None:
            raise RuntimeError(
                f"No actuatorfrcrange for actuator '{actuator.name}'. "
                "Cannot derive a torque limit; refusing to build an "
                "unlimited-torque model."
            )
        actuator.set_to_motor()
        actuator.inheritrange = 0        # position-servo-only convenience
        actuator.biasprm = [0.0] * len(actuator.biasprm)  # clear kp/kv leftovers
        actuator.ctrlrange = frc
        actuator.ctrllimited = 1

    # The stand keyframe's ctrl holds position targets — meaningless (and
    # dangerous) as torques. Keep the pose, drop the command.
    for key in spec.keys:
        if len(key.ctrl):
            key.ctrl = [0.0] * len(key.ctrl)

    if timestep is not None:
        spec.option.timestep = timestep

    if with_floor:
        spec.add_texture(
            name="groundplane",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            width=300,
            height=300,
        )
        material = spec.add_material(name="groundplane", texrepeat=[5, 5], reflectance=0.2)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"

        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0.0, 0.0, 0.05]
        floor.material = "groundplane"
        floor.condim = 3
        floor.friction = [0.8, 0.005, 0.0001]

        light = spec.worldbody.add_light()
        light.pos = [0.0, 0.0, 3.0]
        light.dir = [0.0, 0.0, -1.0]
        # MuJoCo >= 3.11 replaced the boolean `directional` flag with a light
        # type enum; fall back for older releases.
        if hasattr(light, "type"):
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        else:  # pragma: no cover - MuJoCo < 3.11
            light.directional = True

    return spec


def compile_g1_torque_model(
    g1_mjcf_path: Path,
    with_floor: bool = True,
    timestep: float | None = None,
) -> mujoco.MjModel:
    """Build and compile the torque-actuated G1 model."""
    return build_g1_torque_spec(
        g1_mjcf_path, with_floor=with_floor, timestep=timestep
    ).compile()


def torque_limits(mj_model: mujoco.MjModel) -> np.ndarray:
    """Return the per-actuator torque limits [N·m], shape (nu, 2)."""
    return np.asarray(mj_model.actuator_ctrlrange, dtype=float).copy()


def export_xml(g1_mjcf_path: Path, out_path: Path) -> Path:
    """Write a compiled snapshot of the torque model for inspection.

    The snapshot is documentation, not the runtime path: it embeds absolute
    mesh references from the (gitignored) Menagerie clone, so it is not
    portable. Runtime always rebuilds from the spec.
    """
    spec = build_g1_torque_spec(g1_mjcf_path)
    spec.compile()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(spec.to_xml())
    return out_path
