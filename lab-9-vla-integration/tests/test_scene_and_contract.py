"""Lab 9 — M0 tests: the scene, the cameras, and the observation/action contract.

These test the things a demonstration set is silently built on top of. A wrong
state layout or a lossy action codec does not raise; it produces a dataset that
trains a policy to do the wrong thing, and the first sign is a bad success rate
six hours later.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mujoco  # noqa: E402

from lab9_common import (  # noqa: E402
    CAMERAS,
    DT,
    IMAGE_SIZE,
    NU,
    OBJECT_NAMES,
    STATE_DIM,
    TASK_NAMES,
    all_instructions,
    instruction_label,
)
from observations import (  # noqa: E402
    JOINT_ACTION_DIM,
    TASK_ACTION_DIM,
    ObservationRenderer,
    build_state,
    decode_task_action,
    encode_joint_action,
    encode_task_action,
    pelvis_frame,
    pelvis_to_world,
    world_to_pelvis,
)
from vla_scene import (  # noqa: E402
    MARKER_INSET,
    OBJECT_INSET,
    OBJECT_SEPARATION,
    PEDESTAL_Y,
    PICK_PEDESTAL_HALF,
    Randomisation,
    build_vla_scene,
)


@pytest.fixture(scope="module")
def scene():
    return build_vla_scene(DT, Randomisation.sample(0), target="cup")


class TestScene:
    def test_both_objects_compile_with_freejoints(self, scene):
        assert set(scene.object_bodies) == set(OBJECT_NAMES)
        for name in OBJECT_NAMES:
            body = scene.object_bodies[name]
            joint = scene.model.body_jntadr[body]
            assert joint >= 0, f"{name} has no joint"
            assert scene.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE, (
                f"{name} must be free so its pose is a simulation outcome, "
                "not a commanded value"
            )

    def test_every_object_hand_pair_has_a_weld(self, scene):
        assert len(scene.weld_ids) == len(OBJECT_NAMES) * 2
        for weld_id in scene.weld_ids.values():
            assert scene.model.eq_type[weld_id] == mujoco.mjtEq.mjEQ_WELD
            assert scene.data.eq_active[weld_id] == 0, "welds start open"

    def test_objects_rest_on_the_pedestal(self, scene):
        for name in OBJECT_NAMES:
            position = scene.object_position(name)
            assert position[2] > 0.5, f"{name} fell through the pedestal"
            # Inside the pedestal footprint, with the object's own half-extent
            # clear of the edge — Lab 8 L-M5-j: an object placed over an edge
            # tips off, and the same is true of one spawned there.
            half_x, half_y = PICK_PEDESTAL_HALF
            margin = scene.object_half_x(name)
            assert abs(position[0] - 0.40) < half_x - margin
            assert abs(position[1] - PEDESTAL_Y) < half_y - margin

    def test_objects_are_separated_enough_not_to_be_swept(self, scene):
        gap = abs(
            scene.object_position("cup")[0] - scene.object_position("box")[0]
        )
        assert gap > 0.10, (
            "objects closer than ~0.10 m let the forearm sweep the distractor "
            "off the pedestal on the way past"
        )

    def test_marker_sits_further_out_than_the_objects(self):
        assert MARKER_INSET < OBJECT_INSET, (
            "the marker must be inboard of the pedestal's inner edge by more "
            "than an object half-extent (Lab 8 L-M5-j)"
        )

    def test_pedestal_clears_the_hip_line(self):
        inner_face = PEDESTAL_Y + PICK_PEDESTAL_HALF[1]
        # Lab 8 L-M5-f: an inner face at y = -0.22 collides with
        # right_hip_roll_link and fells a controller that walks 12 steps.
        assert inner_face < -0.28, f"inner face at {inner_face:.3f} is too close"

    def test_both_cameras_exist_and_render(self, scene):
        assert set(scene.camera_ids) == set(CAMERAS)
        with ObservationRenderer(scene.model, size=IMAGE_SIZE) as renderer:
            for camera in CAMERAS:
                image = renderer.render(scene.data, camera)
                assert image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
                assert image.dtype == np.uint8
                assert image.std() > 1.0, f"{camera} rendered a flat image"

    def test_target_selection_switches_the_welds(self, scene):
        scene.set_target("box")
        assert np.allclose(scene.payload_position(), scene.object_position("box"))
        scene.set_target("cup")
        assert np.allclose(scene.payload_position(), scene.object_position("cup"))

    def test_scene_carries_no_phantom_mass(self, scene):
        pedestal = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_BODY, "pick_pedestal"
        )
        marker = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_BODY, "drop_marker"
        )
        assert scene.model.body_mass[pedestal] == 0.0
        assert scene.model.body_mass[marker] == 0.0


class TestRandomisation:
    def test_reproducible(self):
        first = Randomisation.sample(7)
        second = Randomisation.sample(7)
        assert first.near_object == second.near_object
        for name in OBJECT_NAMES:
            assert np.allclose(first.offsets[name], second.offsets[name])

    def test_wide_range_is_wider(self):
        widths = []
        for wide in (False, True):
            spread = np.array(
                [
                    Randomisation.sample(seed, wide=wide).offsets["cup"]
                    for seed in range(64)
                ]
            )
            widths.append(np.abs(spread).max())
        assert widths[1] > widths[0], "the wide range must exceed the training one"

    def test_near_object_actually_varies(self):
        chosen = {Randomisation.sample(seed).near_object for seed in range(32)}
        assert chosen == set(OBJECT_NAMES), (
            "if one object were always nearer, the instruction could be ignored"
        )

    def test_object_order_follows_near_object(self):
        for seed in range(8):
            randomisation = Randomisation.sample(seed)
            near = randomisation.object_xy(randomisation.near_object)[0]
            far_name = next(
                n for n in OBJECT_NAMES if n != randomisation.near_object
            )
            assert near < randomisation.object_xy(far_name)[0]

    def test_separation_survives_jitter(self):
        for seed in range(64):
            randomisation = Randomisation.sample(seed)
            gap = abs(
                randomisation.object_xy("cup")[0] - randomisation.object_xy("box")[0]
            )
            assert gap > OBJECT_SEPARATION, "jitter must not close the gap"


class TestObservationContract:
    def test_state_shape_and_finiteness(self, scene):
        state = build_state(scene.data, grasped=False)
        assert state.shape == (STATE_DIM,)
        assert state.dtype == np.float32
        assert np.isfinite(state).all()

    def test_state_excludes_base_position(self, scene):
        """The policy must not be handed its own world x, y or yaw.

        With them it can dead-reckon every task in this lab and ignore both the
        camera and the instruction, and the evaluation would then measure
        nothing. This test is the guard on that decision.
        """
        before = build_state(scene.data, grasped=False)
        data = mujoco.MjData(scene.model)
        data.qpos[:] = scene.data.qpos
        data.qvel[:] = scene.data.qvel
        data.qpos[0] += 3.0     # translate the whole robot in x
        data.qpos[1] -= 2.0     # and in y
        mujoco.mj_forward(scene.model, data)
        after = build_state(data, grasped=False)
        assert np.allclose(before, after), (
            "the state changed when only the base's world position moved"
        )

    def test_grasp_bit_is_carried(self, scene):
        open_state = build_state(scene.data, grasped=False)
        closed_state = build_state(scene.data, grasped=True)
        assert open_state[-1] == 0.0 and closed_state[-1] == 1.0
        assert np.allclose(open_state[:-1], closed_state[:-1])

    def test_joint_block_matches_qpos(self, scene):
        state = build_state(scene.data, grasped=False)
        assert np.allclose(state[:NU], scene.data.qpos[7 : 7 + NU], atol=1e-6)


class TestActionContract:
    def test_task_action_round_trip(self, scene):
        position, yaw = pelvis_frame(scene.data)
        rng = np.random.default_rng(0)
        for _ in range(50):
            right, left = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
            gait, grasp_r, grasp_l = rng.integers(0, 2, 3).astype(float)
            action = encode_task_action(
                right, left, gait, grasp_r, grasp_l, position, yaw
            )
            decoded = decode_task_action(action, position, yaw)
            assert np.allclose(decoded.right_hand, right, atol=1e-6)
            assert np.allclose(decoded.left_hand, left, atol=1e-6)
            assert decoded.gait == gait
            assert decoded.grasp_right == grasp_r
            assert decoded.grasp_left == grasp_l

    def test_action_is_translation_invariant_along_the_walk(self, scene):
        """The same reach must encode identically wherever the robot stands."""
        position, yaw = pelvis_frame(scene.data)
        hand = position + np.array([0.3, -0.2, -0.1])
        first = encode_task_action(hand, hand, 0, 0, 0, position, yaw)
        shifted = position + np.array([0.5, 0.0, 0.0])
        second = encode_task_action(
            hand + np.array([0.5, 0.0, 0.0]), hand + np.array([0.5, 0.0, 0.0]),
            0, 0, 0, shifted, yaw,
        )
        assert np.allclose(first, second, atol=1e-6)

    def test_pelvis_frame_round_trip(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            point = rng.uniform(-2, 2, 3)
            origin = rng.uniform(-2, 2, 3)
            yaw = rng.uniform(-np.pi, np.pi)
            local = world_to_pelvis(point, origin, yaw)
            assert np.allclose(pelvis_to_world(local, origin, yaw), point, atol=1e-9)

    def test_joint_action_dimensions(self, scene):
        action = encode_joint_action(scene.data)
        assert action.shape == (JOINT_ACTION_DIM,) == (NU,)
        assert action.dtype == np.float32

    def test_action_dims_declared_correctly(self):
        assert TASK_ACTION_DIM == 9
        assert JOINT_ACTION_DIM == NU


class TestInstructions:
    def test_every_task_object_pair_has_paraphrases(self):
        for task in TASK_NAMES:
            for obj in OBJECT_NAMES:
                variants = {instruction_label(task, obj, i) for i in range(3)}
                assert len(variants) == 3, f"{task}/{obj} paraphrases collide"

    def test_instructions_distinguish_the_objects(self):
        """Two objects, so the instruction has to choose between them.

        If the two objects' instructions for a task were identical, a policy
        could satisfy the task without reading the language at all.
        """
        for task in TASK_NAMES:
            assert instruction_label(task, "cup") != instruction_label(task, "box")

    def test_held_out_paraphrases_are_excluded_from_training(self):
        train = set(all_instructions(train_only=True))
        every = set(all_instructions(train_only=False))
        assert train < every, "no paraphrase is held out"
        for task in TASK_NAMES:
            for obj in OBJECT_NAMES:
                assert instruction_label(task, obj, 2) not in train


class TestApproachPolicy:
    def test_step_count_depends_on_the_named_object(self):
        from expert import approach_steps_for

        near = approach_steps_for(0.24, marker_x=0.40)
        far = approach_steps_for(0.56, marker_x=0.40)
        assert near != far, (
            "if both objects needed the same walk, the walk task would carry "
            "no language signal"
        )

    def test_step_count_is_clamped_to_the_validated_range(self):
        from expert import MAX_APPROACH_STEPS, MIN_APPROACH_STEPS, approach_steps_for

        for object_x in np.linspace(-1.0, 3.0, 40):
            steps = approach_steps_for(float(object_x), marker_x=0.40)
            assert MIN_APPROACH_STEPS <= steps <= MAX_APPROACH_STEPS

    def test_step_count_is_monotone_in_distance(self):
        from expert import approach_steps_for

        counts = [approach_steps_for(x, marker_x=0.40) for x in np.linspace(0.2, 0.6, 20)]
        assert all(b >= a for a, b in zip(counts, counts[1:], strict=False))


class TestStandingBudget:
    def test_sequence_stays_inside_the_measured_budget(self):
        """The manipulation must fit in the standing-stability budget.

        `_freeze_balance` pins the DCM target at the value it had when the phase
        began. Measured, this controller diverges after roughly 6-7 s of
        continuous standing while an arm is moving; Lab 8 never exceeded that
        because it walked in between. See tasks/LESSONS.md § L-M0-d.
        """
        import expert

        assert expert.STAND_BUDGET_S <= 6.0, (
            f"standing budget is {expert.STAND_BUDGET_S:.1f} s; measured "
            "divergence sets the ceiling near 6 s"
        )
