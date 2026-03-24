"""Lab 9 — Domain randomization for VLA data collection.

Randomizes object positions, colors, and lighting to produce diverse
training demonstrations. All randomization is applied via direct
MuJoCo model/data modifications.
"""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np


class DomainRandomizer:
    """Applies domain randomization to the VLA tabletop scene.

    Randomizes:
    - Object positions (uniform +-offset_range on x/y)
    - Object colors (hue shift via RGB perturbation)
    - Scene lighting direction and intensity

    All randomization is reproducible when a seed is provided.
    """

    def __init__(
        self,
        position_range: float = 0.10,
        color_range: float = 0.15,
        light_intensity_range: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the domain randomizer.

        Args:
            position_range: Max offset in metres for object x/y positions.
            color_range: Max perturbation to RGB channels.
            light_intensity_range: Max perturbation to light diffuse intensity.
            seed: Random seed for reproducibility. None = non-deterministic.
        """
        self.position_range = position_range
        self.color_range = color_range
        self.light_intensity_range = light_intensity_range
        self.rng = np.random.default_rng(seed)

        # Store original values on first call for relative randomization
        self._original_positions: dict[str, np.ndarray] = {}
        self._original_colors: dict[str, np.ndarray] = {}
        self._original_lights: dict[int, np.ndarray] = {}
        self._initialized: bool = False

    def _store_originals(
        self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> None:
        """Cache original values from the model on first call."""
        if self._initialized:
            return

        # Object body positions (from qpos free joints)
        for name in ["cup", "box", "bottle"]:
            jnt_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint"
            )
            qadr = mj_model.jnt_qposadr[jnt_id]
            self._original_positions[name] = mj_data.qpos[qadr : qadr + 3].copy()

        # Object geom colors
        for geom_name in [
            "cup_body", "cup_bottom",
            "box_body",
            "bottle_lower", "bottle_neck", "bottle_cap",
        ]:
            geom_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            if geom_id >= 0:
                self._original_colors[geom_name] = (
                    mj_model.geom_rgba[geom_id].copy()
                )

        # Light diffuse values
        for i in range(mj_model.nlight):
            self._original_lights[i] = mj_model.light_diffuse[i].copy()

        self._initialized = True

    def randomize(
        self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> dict:
        """Apply randomization to the scene.

        Args:
            mj_model: MuJoCo model (modified in-place for colors/lights).
            mj_data: MuJoCo data (modified in-place for positions).

        Returns:
            Dict describing the randomization applied.
        """
        self._store_originals(mj_model, mj_data)
        info: dict = {}

        # --- Object positions ---
        for name in ["cup", "box", "bottle"]:
            jnt_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint"
            )
            qadr = mj_model.jnt_qposadr[jnt_id]
            orig = self._original_positions[name]
            dx = self.rng.uniform(-self.position_range, self.position_range)
            dy = self.rng.uniform(-self.position_range, self.position_range)
            mj_data.qpos[qadr] = orig[0] + dx
            mj_data.qpos[qadr + 1] = orig[1] + dy
            # Keep Z unchanged
            mj_data.qpos[qadr + 2] = orig[2]
            info[f"{name}_offset"] = (dx, dy)

        # --- Object colors ---
        for geom_name, orig_rgba in self._original_colors.items():
            geom_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            if geom_id < 0:
                continue
            perturbation = self.rng.uniform(
                -self.color_range, self.color_range, size=3
            )
            new_rgb = np.clip(orig_rgba[:3] + perturbation, 0.0, 1.0)
            mj_model.geom_rgba[geom_id, :3] = new_rgb
            info[f"{geom_name}_color"] = new_rgb.tolist()

        # --- Lighting ---
        for i, orig_diffuse in self._original_lights.items():
            perturbation = self.rng.uniform(
                -self.light_intensity_range,
                self.light_intensity_range,
                size=3,
            )
            new_diffuse = np.clip(orig_diffuse + perturbation, 0.05, 1.0)
            mj_model.light_diffuse[i] = new_diffuse
            info[f"light_{i}_diffuse"] = new_diffuse.tolist()

        return info

    def reset_seed(self, seed: int) -> None:
        """Reset the random number generator with a new seed.

        Args:
            seed: New random seed.
        """
        self.rng = np.random.default_rng(seed)
