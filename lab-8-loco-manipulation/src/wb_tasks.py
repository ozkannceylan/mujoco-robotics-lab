"""Lab 8 — M1 Step 1.1: whole-body task definitions.

A *task* is a desired behaviour expressed at the velocity level::

    J(q) · q̇  =  ẋ_desired

Each task supplies its Jacobian `J` (m × nv) and the task-space velocity that
would drive its error to zero, `ẋ_des = ẋ_ref + k · e`. The QP (`wb_qp.py`)
then finds the single q̇ that best satisfies all of them at once, ranked by
weight.

Frame conventions (CLAUDE.md Pinocchio Rules, non-negotiable):

* Jacobians are `pin.LOCAL_WORLD_ALIGNED` — translation rows are world-aligned,
  so a positional error expressed in world coordinates maps directly onto them.
* Positions Pinocchio reports live in its own world, 0.793 m below MuJoCo's
  (the pelvis MJCF offset). Task *targets are given in MuJoCo world
  coordinates* and converted internally, so callers never juggle two frames.
* Every Jacobian here is validated against finite differences in
  `tests/test_wb_tasks.py`.

Kinematics are evaluated once per control tick by `TaskStack.update()`, not
per task — with 4+ tasks over a 35-DOF model the redundant FK/Jacobian passes
would otherwise dominate the tick.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pinocchio as pin

from lab8_common import NU, NV, pin_point_to_world, world_point_to_pin

__all__ = [
    "Task",
    "FramePositionTask",
    "FramePoseTask",
    "CoMTask",
    "PostureTask",
    "TaskStack",
]


class Task(ABC):
    """Base class: a weighted velocity-level objective `J q̇ ≈ ẋ_des`.

    Args:
        name: Identifier used in logs and QP diagnostics.
        weight: Relative priority in the QP cost. Levels are separated by
            ≥ 1e2 so a lower task cannot trade away a higher one (see
            `wb_qp.py` for why this lab uses weighted rather than strict
            hierarchical QP).
        gain: Proportional gain converting task error into desired task
            velocity [1/s].
    """

    def __init__(self, name: str, weight: float, gain: float) -> None:
        self.name = name
        self.weight = float(weight)
        self.gain = float(gain)
        self.enabled = True

    @abstractmethod
    def dimension(self) -> int:
        """Number of scalar rows this task contributes."""

    @abstractmethod
    def jacobian(self, model: pin.Model, data: pin.Data, q: np.ndarray) -> np.ndarray:
        """Task Jacobian (m × nv). Assumes kinematics are already computed."""

    @abstractmethod
    def error(self, model: pin.Model, data: pin.Data, q: np.ndarray) -> np.ndarray:
        """Task-space error (m,), target − current, in world coordinates."""

    def desired_velocity(
        self, model: pin.Model, data: pin.Data, q: np.ndarray
    ) -> np.ndarray:
        """Desired task-space velocity `ẋ_des = k · e` (feedforward-free)."""
        return self.gain * self.error(model, data, q)

    # -- acceleration level (used by the inverse-dynamics QP) --------------

    @abstractmethod
    def drift(self, model: pin.Model, data: pin.Data) -> np.ndarray:
        """The `J̇ q̇` term (m,), evaluated with zero joint acceleration."""

    def reference_velocity(self) -> np.ndarray:
        """Feedforward task velocity ẋ_ref (m,). Zero for a static target."""
        return np.zeros(self.dimension())

    def reference_acceleration(self) -> np.ndarray:
        """Feedforward task acceleration ẍ_ref (m,). Zero for a static target."""
        return np.zeros(self.dimension())

    def desired_acceleration(
        self,
        model: pin.Model,
        data: pin.Data,
        q: np.ndarray,
        v: np.ndarray,
        damping_gain: float | None = None,
    ) -> np.ndarray:
        """Task-space PD **plus feedforward**::

            ẍ_des = ẍ_ref + k_p·e + k_d·(ẋ_ref − ẋ)

        The derivative gain defaults to the critically damped `2√k_p`.
        Without the feedforward terms a moving target is tracked with a pure
        lag error — on M1's hand circle that alone was most of the residual
        (18.63 mm RMS → 7.08 mm once ẋ_ref/ẍ_ref were supplied).
        """
        kd = 2.0 * np.sqrt(self.gain) if damping_gain is None else damping_gain
        velocity = self.jacobian(model, data, q) @ v
        return (
            self.reference_acceleration()
            + self.gain * self.error(model, data, q)
            + kd * (self.reference_velocity() - velocity)
        )


class FramePositionTask(Task):
    """Track a 3D world position of a robot frame (e.g. a hand).

    Args:
        frame_name: Pinocchio frame to control.
        target: Desired position in **MuJoCo world** coordinates (3,).
    """

    def __init__(
        self,
        frame_name: str,
        model: pin.Model,
        target: np.ndarray | None = None,
        weight: float = 1e2,
        gain: float = 5.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name or f"pos:{frame_name}", weight, gain)
        if not model.existFrame(frame_name):
            raise ValueError(f"unknown frame '{frame_name}'")
        self.frame_id = model.getFrameId(frame_name)
        self.frame_name = frame_name
        self.target = np.zeros(3) if target is None else np.asarray(target, float).copy()
        self.target_velocity = np.zeros(3)
        self.target_acceleration = np.zeros(3)

    def dimension(self) -> int:
        return 3

    def set_target(
        self,
        target: np.ndarray,
        velocity: np.ndarray | None = None,
        acceleration: np.ndarray | None = None,
    ) -> None:
        """Set the desired world position, optionally with feedforward.

        Supplying the trajectory's own ẋ and ẍ turns a lagging tracker into an
        accurate one; see `Task.desired_acceleration`.
        """
        self.target = np.asarray(target, dtype=float).copy()
        self.target_velocity = (
            np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float).copy()
        )
        self.target_acceleration = (
            np.zeros(3) if acceleration is None else np.asarray(acceleration, dtype=float).copy()
        )

    def reference_velocity(self) -> np.ndarray:
        return self.target_velocity

    def reference_acceleration(self) -> np.ndarray:
        return self.target_acceleration

    def current_position(self, data: pin.Data) -> np.ndarray:
        """Current frame position in MuJoCo world coordinates."""
        return pin_point_to_world(data.oMf[self.frame_id].translation)

    def jacobian(self, model, data, q):
        del q
        return pin.getFrameJacobian(
            model, data, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]

    def error(self, model, data, q):
        del model, q
        return self.target - self.current_position(data)

    def drift(self, model, data):
        return pin.getFrameClassicalAcceleration(
            model, data, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ).linear


class FramePoseTask(Task):
    """Hold or track a full 6D frame pose — used to pin the stance feet.

    The rotational error uses `pin.log3` on the relative rotation, matching the
    LOCAL_WORLD_ALIGNED Jacobian's angular rows.

    Args:
        frame_name: Pinocchio frame to control.
        target_position: Desired position in MuJoCo world coordinates (3,).
        target_rotation: Desired world rotation (3×3).
    """

    def __init__(
        self,
        frame_name: str,
        model: pin.Model,
        target_position: np.ndarray | None = None,
        target_rotation: np.ndarray | None = None,
        weight: float = 1e6,
        gain: float = 10.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name or f"pose:{frame_name}", weight, gain)
        if not model.existFrame(frame_name):
            raise ValueError(f"unknown frame '{frame_name}'")
        self.frame_id = model.getFrameId(frame_name)
        self.frame_name = frame_name
        self.target_position = (
            np.zeros(3) if target_position is None else np.asarray(target_position, float).copy()
        )
        self.target_rotation = (
            np.eye(3) if target_rotation is None else np.asarray(target_rotation, float).copy()
        )

    def dimension(self) -> int:
        return 6

    def capture_current(self, data: pin.Data) -> None:
        """Freeze this task's target at the frame's present pose."""
        placement = data.oMf[self.frame_id]
        self.target_position = pin_point_to_world(placement.translation)
        self.target_rotation = placement.rotation.copy()

    def current_position(self, data: pin.Data) -> np.ndarray:
        return pin_point_to_world(data.oMf[self.frame_id].translation)

    def jacobian(self, model, data, q):
        del q
        return pin.getFrameJacobian(
            model, data, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

    def error(self, model, data, q):
        del model, q
        placement = data.oMf[self.frame_id]
        position_error = self.target_position - pin_point_to_world(placement.translation)
        rotation_error = pin.log3(self.target_rotation @ placement.rotation.T)
        return np.concatenate([position_error, rotation_error])

    def drift(self, model, data):
        acc = pin.getFrameClassicalAcceleration(
            model, data, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return np.concatenate([acc.linear, acc.angular])


class CoMTask(Task):
    """Regulate the centre of mass — the balance task.

    Args:
        target: Desired CoM in MuJoCo world coordinates (3,).
        axes: Which components to control. Standing balance only needs the
            horizontal ones (`(0, 1)`); constraining z as well fights the
            natural knee compliance for no benefit.
    """

    def __init__(
        self,
        model: pin.Model,
        target: np.ndarray | None = None,
        axes: tuple[int, ...] = (0, 1),
        weight: float = 1e4,
        gain: float = 5.0,
        name: str = "com",
    ) -> None:
        del model
        super().__init__(name, weight, gain)
        self.axes = tuple(axes)
        self.target = np.zeros(3) if target is None else np.asarray(target, float).copy()

    def dimension(self) -> int:
        return len(self.axes)

    def set_target(self, target: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=float).copy()

    def current_com(self, data: pin.Data) -> np.ndarray:
        """CoM in MuJoCo world coordinates."""
        return pin_point_to_world(data.com[0])

    def jacobian(self, model, data, q):
        del model, q
        return data.Jcom[list(self.axes), :]

    def error(self, model, data, q):
        del model, q
        return (self.target - self.current_com(data))[list(self.axes)]

    def drift(self, model, data):
        del model
        return data.acom[0][list(self.axes)]


class PostureTask(Task):
    """Pull the actuated joints toward a nominal configuration.

    The lowest-priority task: it resolves the redundancy left over once
    balance, feet and hand are satisfied, and keeps the arms from drifting
    into strange poses. Acts on joints only — the floating base is not
    commanded (and cannot be).
    """

    def __init__(
        self,
        q_nominal: np.ndarray,
        weight: float = 1.0,
        gain: float = 1.0,
        name: str = "posture",
    ) -> None:
        super().__init__(name, weight, gain)
        self.q_nominal = np.asarray(q_nominal, dtype=float).copy()
        if self.q_nominal.shape != (NU,):
            raise ValueError(f"q_nominal must be ({NU},), got {self.q_nominal.shape}")
        self._jacobian = np.hstack([np.zeros((NU, 6)), np.eye(NU)])

    def dimension(self) -> int:
        return NU

    def jacobian(self, model, data, q):
        del model, data, q
        return self._jacobian

    def error(self, model, data, q):
        del model, data
        return self.q_nominal - np.asarray(q)[7:]

    def drift(self, model, data):
        del model, data
        return np.zeros(NU)


class TaskStack:
    """A prioritised collection of tasks sharing one kinematics evaluation.

    Usage per control tick::

        stack.update(q)                 # FK + Jacobians + CoM, once
        J, xdot, weights = stack.assemble(q)
    """

    def __init__(self, model: pin.Model, data: pin.Data, tasks: list[Task] | None = None):
        self.model = model
        self.data = data
        self.tasks: list[Task] = list(tasks or [])

    def add(self, task: Task) -> Task:
        """Append a task and return it (convenient for keeping a handle)."""
        self.tasks.append(task)
        return task

    @property
    def active(self) -> list[Task]:
        return [task for task in self.tasks if task.enabled]

    def update(self, q: np.ndarray) -> None:
        """Evaluate forward kinematics, frame placements, Jacobians and CoM."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.jacobianCenterOfMass(self.model, self.data, q)

    def update_dynamics(self, q: np.ndarray, v: np.ndarray) -> None:
        """Kinematics plus the drift terms the acceleration-level QP needs.

        Running forward kinematics with **zero** acceleration makes every
        frame/CoM acceleration Pinocchio reports equal to the pure `J̇ q̇`
        drift, which is exactly the term the QP must cancel.
        """
        zero_acceleration = np.zeros(self.model.nv)
        pin.forwardKinematics(self.model, self.data, q, v, zero_acceleration)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.jacobianCenterOfMass(self.model, self.data, q)
        pin.centerOfMass(self.model, self.data, q, v, zero_acceleration)

    def assemble(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stack active tasks into (J, ẋ_des, per-row weights).

        Returns:
            J: (M × nv) stacked Jacobian.
            xdot: (M,) stacked desired task velocities.
            weights: (M,) per-row weights, repeated from each task's weight.
        """
        jacobians, velocities, weights = [], [], []
        for task in self.active:
            jacobians.append(task.jacobian(self.model, self.data, q))
            velocities.append(task.desired_velocity(self.model, self.data, q))
            weights.append(np.full(task.dimension(), task.weight))
        if not jacobians:
            return np.zeros((0, NV)), np.zeros(0), np.zeros(0)
        return np.vstack(jacobians), np.concatenate(velocities), np.concatenate(weights)

    def errors(self, q: np.ndarray) -> dict[str, float]:
        """Per-task error norm — for gate tables and QP diagnostics."""
        return {
            task.name: float(np.linalg.norm(task.error(self.model, self.data, q)))
            for task in self.active
        }
