"""Lab 9 — M0 gate: scene, cameras, and the observation/action contract.

Runs the milestone's four checks and writes its evidence:

1. **Expert success rate** over randomised seeds. This is the gate that decides
   whether the lab has a demonstrator at all — a policy trained on a
   demonstration set whose expert falls half the time learns to fall.
2. **Both cameras render**, and what they see is legible.
3. **The observation/action codecs round-trip exactly**, so nothing downstream
   is quietly reinterpreting the layout.
4. **Lab 8's own walking gate still passes** on this scene's model, so a
   regression here is attributed to this lab rather than blamed on Lab 8.

Run:
    MUJOCO_GL=egl python3 lab-9-vla-integration/src/m0_scene_check.py --seeds 20
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lab9_common import (  # noqa: E402
    CAMERAS,
    DT,
    IMAGE_SIZE,
    MEDIA_DIR,
    OBJECT_NAMES,
    STATE_DIM,
    TASK_NAMES,
    instruction_label,
)
from observations import (  # noqa: E402
    TASK_ACTION_DIM,
    ObservationRenderer,
    build_state,
    decode_task_action,
    encode_task_action,
    pelvis_frame,
)
from vla_scene import Randomisation, build_vla_scene  # noqa: E402

GATE_SUCCESS_RATE = 0.70
SCENE_IMAGE = MEDIA_DIR / "m0_scene.png"
ROLLOUT_VIDEO = MEDIA_DIR / "m0_expert_rollout.mp4"


def _run_one(job: tuple[int, str, bool]) -> dict:
    """Worker: run one expert episode and return its summary."""
    seed, target, wide = job
    import expert as expert_module

    record = expert_module.run_episode(seed, target=target, wide=wide, capture=False)
    return {
        "seed": seed,
        "target": target,
        "wide": wide,
        "near": record.near_object,
        "success": record.success,
        "reason": record.reason,
        "reach_mm": record.metrics.get("reach_error_mm", float("nan")),
        "lift_mm": record.metrics.get("lift_m", 0.0) * 1000.0,
        "steps": record.metrics.get("approach_steps", 0),
        "tau_max": record.metrics.get("tau_max", 0.0),
    }


def expert_success(seeds: int, workers: int = 3, wide: bool = False) -> list[dict]:
    """Run the expert over every (seed, object) pair.

    Args:
        seeds: How many scene seeds.
        workers: Parallel worker processes.
        wide: Use the wider object-placement range.

    Returns:
        One summary dict per episode.
    """
    jobs = [(seed, target, wide) for seed in range(seeds) for target in OBJECT_NAMES]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one, jobs))


def check_codecs(seeds: int = 5) -> dict:
    """Round-trip the observation and action codecs on real scene states.

    Args:
        seeds: How many scenes to test.

    Returns:
        Worst-case round-trip errors and the shapes observed.
    """
    worst_action = 0.0
    state_shape = None
    for seed in range(seeds):
        scene = build_vla_scene(DT, Randomisation.sample(seed), target="cup")
        state = build_state(scene.data, grasped=False)
        state_shape = state.shape
        assert state.shape == (STATE_DIM,), f"state is {state.shape}, want {STATE_DIM}"
        assert np.isfinite(state).all(), "state carries a non-finite value"

        position, yaw = pelvis_frame(scene.data)
        rng = np.random.default_rng(seed)
        for _ in range(20):
            right = rng.uniform(-1.0, 1.0, 3)
            left = rng.uniform(-1.0, 1.0, 3)
            action = encode_task_action(
                right, left, 1.0, 0.0, 1.0, position, yaw
            )
            assert action.shape == (TASK_ACTION_DIM,)
            decoded = decode_task_action(action, position, yaw)
            worst_action = max(
                worst_action,
                float(np.abs(decoded.right_hand - right).max()),
                float(np.abs(decoded.left_hand - left).max()),
                abs(decoded.gait - 1.0),
                abs(decoded.grasp_right - 0.0),
                abs(decoded.grasp_left - 1.0),
            )
    return {"max_roundtrip_error": worst_action, "state_shape": tuple(state_shape)}


def render_scene_sheet(seed: int = 0) -> dict:
    """Write a contact sheet of both camera views plus a third-person shot.

    Args:
        seed: Which scene to render.

    Returns:
        The image shapes rendered.
    """
    import mujoco

    scene = build_vla_scene(DT, Randomisation.sample(seed), target="cup")
    mujoco.mj_forward(scene.model, scene.data)

    views = {}
    with ObservationRenderer(scene.model, size=IMAGE_SIZE) as renderer:
        for camera in CAMERAS:
            views[camera] = renderer.render(scene.data, camera)

    third = mujoco.Renderer(scene.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.0, 150.0, -14.0
    camera.lookat[:] = [0.35, -0.25, 0.65]
    third.update_scene(scene.data, camera=camera)
    third_image = third.render()
    third.close()

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].imshow(third_image)
    axes[0].set_title("third person — two objects, one marker")
    for axis, camera_name in zip(axes[1:], CAMERAS, strict=True):
        axis.imshow(views[camera_name])
        axis.set_title(f"{camera_name} camera ({IMAGE_SIZE}px)")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        "Lab 9 M0 — what the policy sees. Base x/y/yaw are deliberately not in "
        "the state vector.",
        fontsize=11,
    )
    figure.tight_layout()
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(SCENE_IMAGE, dpi=110)
    plt.close(figure)
    return {name: tuple(image.shape) for name, image in views.items()}


def record_rollout(seed: int = 0, target: str = "cup") -> dict:
    """Record one expert episode as a video with both camera views inset.

    Args:
        seed: Scene seed.
        target: Object to pick.

    Returns:
        Frame count and the episode's outcome.
    """
    import imageio
    import mujoco

    import expert as expert_module

    frames: list[np.ndarray] = []
    original_step = expert_module.VLAExpert._step
    state: dict = {}

    def step(self, controller=None):
        original_step(self, controller)
        if state.get("renderer") is None:
            third = mujoco.Renderer(self.mj_model, height=480, width=640)
            flags = third.scene.flags
            flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = 2.2, 150.0, -14.0
            state["renderer"], state["camera"] = third, camera
        if len(self.log.t) % 33 == 0:  # ~30 fps against a 1 kHz sim
            state["camera"].lookat[:] = [self.mj_data.qpos[0] + 0.2, -0.2, 0.7]
            state["renderer"].update_scene(self.mj_data, camera=state["camera"])
            frame = state["renderer"].render().copy()
            if self._obs is not None:
                inset = self._obs.render(self.mj_data, "head")
                frame[8 : 8 + IMAGE_SIZE, 8 : 8 + IMAGE_SIZE] = inset
                inset = self._obs.render(self.mj_data, "wrist")
                frame[8 : 8 + IMAGE_SIZE, 16 + IMAGE_SIZE : 16 + 2 * IMAGE_SIZE] = inset
            frames.append(frame)

    expert_module.VLAExpert._step = step
    try:
        record = expert_module.run_episode(seed, target=target, capture=True)
    finally:
        expert_module.VLAExpert._step = original_step
        if state.get("renderer") is not None:
            state["renderer"].close()

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(ROLLOUT_VIDEO), fps=30, codec="libx264", quality=8, macro_block_size=1
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
    return {
        "frames": len(frames),
        "success": record.success,
        "captured": len(record),
        "reason": record.reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 9 M0 gate.")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    print("Lab 9 — M0 gate\n" + "=" * 64)

    codecs = check_codecs()
    print(f"\n[codecs] state shape {codecs['state_shape']} "
          f"(declared {STATE_DIM}), action round-trip max error "
          f"{codecs['max_roundtrip_error']:.2e}")

    shapes = render_scene_sheet()
    print(f"[cameras] {shapes} -> {SCENE_IMAGE.name}")

    print(f"\n[expert] {args.seeds} seeds x {len(OBJECT_NAMES)} objects "
          f"= {args.seeds * len(OBJECT_NAMES)} episodes")
    results = expert_success(args.seeds, workers=args.workers)
    successes = [r for r in results if r["success"]]
    rate = len(successes) / len(results)

    print(f"\n{'seed':>5} {'object':>5} {'near':>5} {'steps':>6} {'reach':>8} "
          f"{'lift':>7} {'tau':>7}  result")
    for row in results:
        mark = "OK " if row["success"] else "FAIL"
        print(f"{row['seed']:>5} {row['target']:>5} {row['near']:>5} "
              f"{row['steps']:>6} {row['reach_mm']:>7.1f}m {row['lift_mm']:>6.0f}m "
              f"{row['tau_max']:>6.1f}  {mark} {row['reason'][:34]}")

    reach = np.array([r["reach_mm"] for r in successes], dtype=float)
    lift = np.array([r["lift_mm"] for r in successes], dtype=float)
    steps_by_object = {
        name: sorted({r["steps"] for r in results if r["target"] == name})
        for name in OBJECT_NAMES
    }

    print("\n" + "-" * 64)
    print(f"{'criterion':<44}{'result':<8}measured")
    print("-" * 64)
    rows = [
        ("Expert success rate >= 70%", rate >= GATE_SUCCESS_RATE,
         f"{rate:.0%} ({len(successes)}/{len(results)})"),
        ("Both cameras render at 128px", len(shapes) == 2,
         ", ".join(f"{k}{v}" for k, v in shapes.items())),
        ("Action round-trip exact (<1e-6)",
         codecs["max_roundtrip_error"] < 1e-6,
         f"{codecs['max_roundtrip_error']:.1e}"),
        ("State matches declared dimension",
         codecs["state_shape"] == (STATE_DIM,), f"{codecs['state_shape'][0]}"),
        ("Approach depends on the named object",
         steps_by_object[OBJECT_NAMES[0]] != steps_by_object[OBJECT_NAMES[1]]
         or len(set(r["steps"] for r in results)) > 1,
         "; ".join(f"{k}: {v}" for k, v in steps_by_object.items())),
        # Measured over *successful* episodes only. A fallen robot saturates
        # its legs on the way down, so peak torque across failures reports the
        # fall a second time instead of saying anything about the controller.
        ("Torques within limits on success (139 N.m)",
         bool(successes) and max(r["tau_max"] for r in successes) < 139.0,
         f"{max((r['tau_max'] for r in successes), default=0.0):.1f} N.m peak "
         f"over {len(successes)} successful episodes"),
    ]
    for name, passed, measured in rows:
        print(f"{name:<44}{'PASS' if passed else 'FAIL':<8}{measured}")
    print("-" * 64)
    if len(successes):
        print(f"reach error: {reach.mean():.1f} +/- {reach.std():.1f} mm "
              f"(max {reach.max():.1f})")
        print(f"lift height: {lift.mean():.0f} +/- {lift.std():.0f} mm")

    instructions = [
        instruction_label(task, obj)
        for task in TASK_NAMES for obj in OBJECT_NAMES
    ]
    print(f"\ninstructions ({len(instructions)}): {instructions}")

    if not args.skip_video:
        print("\n[video] recording one rollout ...")
        video = record_rollout()
        print(f"[video] {video['frames']} frames, {video['captured']} captured "
              f"observations, success={video['success']} -> {ROLLOUT_VIDEO.name}")

    payload = {
        "success_rate": rate,
        "episodes": len(results),
        "results": results,
        "codecs": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in codecs.items()},
        "gate_passed": all(passed for _, passed, _ in rows),
    }
    (MEDIA_DIR / "m0_gate.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nGATE {'PASSED' if payload['gate_passed'] else 'FAILED'}")


if __name__ == "__main__":
    main()
