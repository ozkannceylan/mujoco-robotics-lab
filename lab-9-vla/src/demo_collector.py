"""Lab 9 — Expert demonstration collector.

Generates IK-based expert demonstrations for tabletop manipulation tasks.
Demonstrations are recorded as trajectories containing camera images,
proprioception, actions, and language instructions, then saved as NumPy npz.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from lab9_common import (
    ACTION_CHUNK_SIZE,
    ACTION_DIM,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DEMOS_DIR,
    HEAD_CAM,
    NUM_ROBOT_JOINTS,
    OBJECT_NAMES,
    POLICY_DT,
    PROPRIO_DIM,
    RIGHT_ARM_JOINT_INDICES,
    ROBOT_QPOS_END,
    ROBOT_QPOS_START,
    SIM_DT,
    STEPS_PER_POLICY,
    TABLE_SURFACE_Z,
    WRIST_CAM,
    CameraRenderer,
    TaskDefinition,
    get_ee_position,
    get_object_position,
    get_proprioception,
    load_mujoco_model,
    set_grasp_weld,
)


@dataclasses.dataclass
class DemoTrajectory:
    """Container for a single demonstration trajectory."""

    task_name: str
    language: str
    wrist_images: list[np.ndarray]      # list of (H, W, 3) uint8
    head_images: list[np.ndarray]       # list of (H, W, 3) uint8
    proprioception: list[np.ndarray]    # list of (PROPRIO_DIM,) float64
    actions: list[np.ndarray]           # list of (ACTION_DIM,) float64
    success: bool = False

    @property
    def length(self) -> int:
        """Number of timesteps."""
        return len(self.actions)

    def save(self, filepath: Path) -> None:
        """Save trajectory to a compressed npz file.

        Args:
            filepath: Destination .npz path.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Downsample images to save space: store at reduced resolution
        wrist_stack = np.stack(self.wrist_images, axis=0)   # (T, H, W, 3)
        head_stack = np.stack(self.head_images, axis=0)
        proprio_stack = np.stack(self.proprioception, axis=0)
        action_stack = np.stack(self.actions, axis=0)
        np.savez_compressed(
            filepath,
            task_name=np.array(self.task_name),
            language=np.array(self.language),
            wrist_images=wrist_stack,
            head_images=head_stack,
            proprioception=proprio_stack,
            actions=action_stack,
            success=np.array(self.success),
        )

    @classmethod
    def load(cls, filepath: Path) -> "DemoTrajectory":
        """Load trajectory from npz file.

        Args:
            filepath: Source .npz path.

        Returns:
            Reconstructed DemoTrajectory.
        """
        data = np.load(filepath, allow_pickle=True)
        wrist_imgs = list(data["wrist_images"])
        head_imgs = list(data["head_images"])
        proprios = list(data["proprioception"])
        acts = list(data["actions"])
        return cls(
            task_name=str(data["task_name"]),
            language=str(data["language"]),
            wrist_images=wrist_imgs,
            head_images=head_imgs,
            proprioception=proprios,
            actions=acts,
            success=bool(data["success"]),
        )


class DemoCollector:
    """Generates expert demonstrations via IK-based trajectory planning.

    The expert controller uses a simple pick-and-place state machine:
    1. Move hand above the object (approach)
    2. Lower hand to grasp height
    3. Close gripper / activate weld
    4. Lift object
    5. Move to target position (for move tasks)
    6. Open gripper / deactivate weld
    """

    def __init__(
        self,
        image_width: int = CAMERA_WIDTH,
        image_height: int = CAMERA_HEIGHT,
        max_episode_steps: int = 200,
    ) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.max_episode_steps = max_episode_steps
        self._mj_model: Optional[mujoco.MjModel] = None
        self._mj_data: Optional[mujoco.MjData] = None
        self._renderer: Optional[CameraRenderer] = None

    def _init_sim(self) -> None:
        """Initialize or reset the simulation."""
        self._mj_model, self._mj_data = load_mujoco_model()
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = CameraRenderer(
            self._mj_model,
            width=self.image_width,
            height=self.image_height,
        )
        # Step to settle initial state
        mujoco.mj_forward(self._mj_model, self._mj_data)

    def _generate_waypoints(
        self, task: TaskDefinition
    ) -> list[dict]:
        """Generate waypoints for a pick / pick-and-place task.

        Args:
            task: The task definition.

        Returns:
            List of waypoint dicts with keys:
              - 'ee_target': (3,) target end-effector position
              - 'gripper': float gripper command (0=open, 0.08=closed)
              - 'grasp_weld': bool whether to activate weld
              - 'steps': int number of policy steps to hold
        """
        assert self._mj_model is not None and self._mj_data is not None
        obj_pos = get_object_position(self._mj_model, self._mj_data, task.object_name)

        approach_z = obj_pos[2] + 0.12
        grasp_z = obj_pos[2] + 0.02
        lift_z = obj_pos[2] + task.grasp_height

        waypoints = [
            # 1. Approach above object
            {
                "ee_target": np.array([obj_pos[0], obj_pos[1], approach_z]),
                "gripper": 0.0,
                "grasp_weld": False,
                "steps": 30,
            },
            # 2. Lower to grasp
            {
                "ee_target": np.array([obj_pos[0], obj_pos[1], grasp_z]),
                "gripper": 0.0,
                "grasp_weld": False,
                "steps": 20,
            },
            # 3. Close gripper
            {
                "ee_target": np.array([obj_pos[0], obj_pos[1], grasp_z]),
                "gripper": 0.08,
                "grasp_weld": True,
                "steps": 10,
            },
            # 4. Lift
            {
                "ee_target": np.array([obj_pos[0], obj_pos[1], lift_z]),
                "gripper": 0.08,
                "grasp_weld": True,
                "steps": 25,
            },
        ]

        # For move tasks, add transport and release
        if "move" in task.name:
            waypoints.extend([
                # 5. Transport to target
                {
                    "ee_target": np.array([
                        task.target_pos[0],
                        task.target_pos[1],
                        lift_z,
                    ]),
                    "gripper": 0.08,
                    "grasp_weld": True,
                    "steps": 30,
                },
                # 6. Lower to target
                {
                    "ee_target": task.target_pos.copy(),
                    "gripper": 0.08,
                    "grasp_weld": True,
                    "steps": 20,
                },
                # 7. Release
                {
                    "ee_target": task.target_pos.copy(),
                    "gripper": 0.0,
                    "grasp_weld": False,
                    "steps": 10,
                },
            ])

        return waypoints

    def _simple_ik_step(
        self,
        ee_target: np.ndarray,
        gripper_cmd: float,
    ) -> np.ndarray:
        """Compute a joint position target via Jacobian-based IK.

        Uses a damped least-squares approach on the right arm only.
        Leg and left arm joints hold their current positions.

        Args:
            ee_target: Desired (3,) end-effector position.
            gripper_cmd: Gripper opening (0=open, 0.08=closed).

        Returns:
            Full action vector (ACTION_DIM,) of joint position targets.
        """
        assert self._mj_model is not None and self._mj_data is not None
        model = self._mj_model
        data = self._mj_data

        # Current action = current joint positions (hold everything)
        action = data.qpos[ROBOT_QPOS_START:ROBOT_QPOS_END].copy()

        # Get current EE position
        ee_pos = get_ee_position(model, data, "right_ee")
        error = ee_target - ee_pos

        # Compute Jacobian for right_ee site
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_ee")
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site_id)

        # Extract columns for right arm joints (velocity indices)
        # Right arm velocity indices: 6 (root dof) + joint indices
        right_arm_vel_indices = [6 + idx for idx in RIGHT_ARM_JOINT_INDICES]
        J = jacp[:, right_arm_vel_indices]  # (3, 7)

        # Damped least-squares
        damping = 0.01
        JtJ = J.T @ J + damping * np.eye(J.shape[1])
        dq = np.linalg.solve(JtJ, J.T @ error)

        # Apply to right arm joints
        for i, joint_idx in enumerate(RIGHT_ARM_JOINT_INDICES):
            action[joint_idx] += dq[i] * 0.3  # gain

        # Set gripper
        action[RIGHT_ARM_JOINT_INDICES[-1]] = gripper_cmd

        return action

    def collect_demo(
        self,
        task: TaskDefinition,
        randomizer: Optional[object] = None,
    ) -> DemoTrajectory:
        """Collect a single expert demonstration for the given task.

        Args:
            task: Task definition specifying what to do.
            randomizer: Optional DomainRandomizer to apply before episode.

        Returns:
            A DemoTrajectory with recorded observations and actions.
        """
        self._init_sim()
        assert self._mj_model is not None and self._mj_data is not None
        assert self._renderer is not None

        # Apply domain randomization if provided
        if randomizer is not None:
            randomizer.randomize(self._mj_model, self._mj_data)
            mujoco.mj_forward(self._mj_model, self._mj_data)

        # Stabilize the robot for a few steps
        for _ in range(100):
            mujoco.mj_step(self._mj_model, self._mj_data)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        # Generate waypoints
        waypoints = self._generate_waypoints(task)

        trajectory = DemoTrajectory(
            task_name=task.name,
            language=task.language,
            wrist_images=[],
            head_images=[],
            proprioception=[],
            actions=[],
        )

        step_count = 0
        for wp in waypoints:
            ee_target = wp["ee_target"]
            gripper_cmd = wp["gripper"]
            grasp_weld = wp["grasp_weld"]
            num_steps = wp["steps"]

            # Set weld state
            set_grasp_weld(
                self._mj_model, self._mj_data, task.object_name, grasp_weld
            )

            for _ in range(num_steps):
                if step_count >= self.max_episode_steps:
                    break

                # Compute action
                action = self._simple_ik_step(ee_target, gripper_cmd)

                # Record observations before stepping
                mujoco.mj_forward(self._mj_model, self._mj_data)
                wrist_img = self._renderer.render(self._mj_data, WRIST_CAM)
                head_img = self._renderer.render(self._mj_data, HEAD_CAM)
                proprio = get_proprioception(self._mj_data)

                trajectory.wrist_images.append(wrist_img)
                trajectory.head_images.append(head_img)
                trajectory.proprioception.append(proprio)
                trajectory.actions.append(action)

                # Apply action and step simulation
                self._mj_data.ctrl[:] = action
                for _ in range(STEPS_PER_POLICY):
                    mujoco.mj_step(self._mj_model, self._mj_data)

                step_count += 1

        # Check success
        mujoco.mj_forward(self._mj_model, self._mj_data)
        obj_pos = get_object_position(
            self._mj_model, self._mj_data, task.object_name
        )
        # For pick tasks: success if object is lifted above table
        if "pick" in task.name:
            trajectory.success = obj_pos[2] > TABLE_SURFACE_Z + 0.08
        else:
            trajectory.success = task.check_success(obj_pos)

        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        return trajectory

    def collect_and_save(
        self,
        task: TaskDefinition,
        demo_id: int,
        randomizer: Optional[object] = None,
    ) -> tuple[bool, Path]:
        """Collect a demo and save it to disk.

        Args:
            task: Task definition.
            demo_id: Integer identifier for this demo.
            randomizer: Optional domain randomizer.

        Returns:
            (success, filepath) tuple.
        """
        traj = self.collect_demo(task, randomizer=randomizer)
        task_dir = DEMOS_DIR / task.name
        filepath = task_dir / f"demo_{demo_id:04d}.npz"
        traj.save(filepath)
        return traj.success, filepath
