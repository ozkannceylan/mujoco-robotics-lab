"""Walking controller for the G1 humanoid.

Combines footstep planning, LIPM-based CoM planning, and whole-body IK
to generate walking motions executed via MuJoCo position servos.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from lab7_common import (
    ACTUATOR_KP,
    NUM_ACTUATED,
    NQ_FREEFLYER,
    NV_FREEFLYER,
    NQ_PIN,
    Q_HOME,
    DT,
    HIP_WIDTH,
    get_state_from_mujoco,
    get_mj_joint_dofadr,
    build_q_pin,
)
from g1_model import G1Model
from footstep_planner import FootstepPlanner, Footstep
from lipm_planner import LIPMPlanner


class WalkingController:
    """Whole-body walking controller using LIPM + IK.

    Workflow:
      1. Plan footsteps (FootstepPlanner).
      2. Generate ZMP reference from footsteps.
      3. Generate CoM trajectory via LIPM preview control.
      4. At each timestep, compute desired foot poses and CoM.
      5. Solve whole-body IK for desired joint positions.
      6. Send to position servos with gravity feedforward.
    """

    def __init__(
        self,
        g1: G1Model,
        step_length: float = 0.15,
        step_height: float = 0.04,
        step_duration: float = 0.5,
        double_support_duration: float = 0.2,
        com_height: float = 0.55,
    ) -> None:
        """Initialize walking controller.

        Args:
            g1: G1 Pinocchio model wrapper.
            step_length: Forward step length [m].
            step_height: Swing foot apex height [m].
            step_duration: Single support duration [s].
            double_support_duration: Double support duration [s].
            com_height: Desired CoM height [m].
        """
        self.g1 = g1
        self.com_height = com_height

        self.footstep_planner = FootstepPlanner(
            step_length=step_length,
            step_width=2 * HIP_WIDTH,
            step_height=step_height,
            step_duration=step_duration,
            double_support_duration=double_support_duration,
        )

        self.lipm = LIPMPlanner(
            z_c=com_height,
            dt=DT,
            preview_steps=150,
        )

        # Planned trajectories (set after plan())
        self.steps: List[Footstep] = []
        self.com_traj_x: Optional[np.ndarray] = None
        self.com_traj_y: Optional[np.ndarray] = None
        self.total_time: float = 0.0
        self.total_steps_count: int = 0

        # Cache
        self._dof_addrs = None
        self._q_prev: Optional[np.ndarray] = None

    def plan(self, num_steps: int = 10) -> float:
        """Plan a walking sequence.

        Args:
            num_steps: Number of footsteps.

        Returns:
            Total trajectory time [s].
        """
        self.total_steps_count = num_steps
        self.steps = self.footstep_planner.plan_footsteps(
            num_steps=num_steps, first_foot="right"
        )

        # Compute total time
        self.total_time = (
            self.steps[-1].t_end
            + self.footstep_planner.double_support_duration
            + 1.0  # settling time
        )

        # Generate ZMP reference
        t_arr, zmp_ref_x, zmp_ref_y = self.footstep_planner.generate_zmp_reference(
            self.steps, total_time=self.total_time
        )

        # Generate CoM trajectory via LIPM
        self.com_traj_x, self.com_traj_y, self.zmp_x, self.zmp_y = (
            self.lipm.plan_com_trajectory(zmp_ref_x, zmp_ref_y)
        )

        self._q_prev = None
        return self.total_time

    def get_time_index(self, t: float) -> int:
        """Convert time to trajectory index.

        Args:
            t: Current time [s].

        Returns:
            Trajectory index, clamped to valid range.
        """
        idx = int(t / DT)
        N = len(self.com_traj_x) if self.com_traj_x is not None else 0
        return min(max(idx, 0), N - 1)

    def compute_control(
        self,
        mj_model,
        mj_data,
        t: float,
    ) -> np.ndarray:
        """Compute walking control for the current timestep.

        Args:
            mj_model: MuJoCo model.
            mj_data: MuJoCo data.
            t: Current simulation time [s].

        Returns:
            Control signal array [NUM_ACTUATED].
        """
        if self._dof_addrs is None:
            self._dof_addrs = get_mj_joint_dofadr(mj_model)

        # Get current state
        q_pin, v_pin = get_state_from_mujoco(mj_model, mj_data)

        # Get trajectory index
        idx = self.get_time_index(t)

        # Desired CoM position
        com_des = np.array([
            self.com_traj_x[idx],
            self.com_traj_y[idx],
            self.com_height,
        ])

        # Get desired foot positions
        left_foot_des, right_foot_des, phase = (
            self.footstep_planner.get_foot_positions_at_time(t, self.steps)
        )

        # Desired foot rotations (flat, identity)
        R_flat = np.eye(3)

        # Use previous IK solution as warm start
        q_init = self._q_prev if self._q_prev is not None else q_pin.copy()

        # Update base position from current state for IK init
        q_init[0:3] = q_pin[0:3]
        q_init[3:7] = q_pin[3:7]

        # Solve whole-body IK
        q_des = self.g1.whole_body_ik(
            q_current=q_init,
            com_des=com_des,
            left_foot_des=left_foot_des,
            right_foot_des=right_foot_des,
            left_foot_rot=R_flat,
            right_foot_rot=R_flat,
            posture_ref=Q_HOME,
            max_iter=10,
            dt_ik=0.05,
            w_com=5.0,
            w_foot=100.0,
            w_foot_rot=30.0,
            w_posture=0.05,
            damping=1e-4,
        )

        self._q_prev = q_des.copy()

        # Extract desired joint positions
        q_joints_des = q_des[NQ_FREEFLYER:]

        # Build control with gravity feedforward
        ctrl = np.zeros(NUM_ACTUATED)
        for i in range(NUM_ACTUATED):
            bias = mj_data.qfrc_bias[self._dof_addrs[i]]
            ctrl[i] = q_joints_des[i] + bias / ACTUATOR_KP[i]

        return ctrl

    def get_planned_data(self) -> Dict[str, np.ndarray]:
        """Get planned trajectory data for plotting.

        Returns:
            Dictionary with planned trajectory arrays.
        """
        N = len(self.com_traj_x) if self.com_traj_x is not None else 0
        t = np.arange(N) * DT
        return {
            "time": t,
            "com_x": self.com_traj_x,
            "com_y": self.com_traj_y,
            "zmp_x": self.zmp_x,
            "zmp_y": self.zmp_y,
        }
