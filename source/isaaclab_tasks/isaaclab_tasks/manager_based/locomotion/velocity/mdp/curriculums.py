# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def modify_reward_weight(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int
) -> float:
    """Curriculum that modifies a reward weight after a fixed number of training steps.

    This is useful for gradually introducing penalty terms or increasing task rewards
    once the robot has learned basic locomotion.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term to modify.
        weight: The new weight to apply to the reward term.
        num_steps: The number of steps after which the weight change is applied.

    Returns:
        The current weight of the reward term.
    """
    if env.common_step_counter > num_steps:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)
    return weight


def modify_push_force(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    max_velocity: Sequence[float],
    interval: int,
    starting_step: float = 0.0,
) -> float:
    """Curriculum that progressively increases the push (perturbation) force on the robot.

    The push force increases when the robot rarely falls, and decreases when it falls too often.
    This creates an adaptive disturbance curriculum.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the push event term.
        max_velocity: Maximum [x, y] push velocity limits.
        interval: The number of steps between curriculum updates.
        starting_step: The step count at which to begin applying the curriculum.

    Returns:
        The current maximum push velocity setting.
    """
    try:
        term_cfg = env.event_manager.get_term_cfg("push_robot")
    except Exception:
        return 0.0

    curr_setting = term_cfg.params["velocity_range"]["x"][1]

    if env.common_step_counter < starting_step:
        return curr_setting

    if env.common_step_counter % interval == 0:
        base_contacts = torch.sum(env.termination_manager._term_dones["base_contact"])
        timeouts = torch.sum(env.termination_manager._term_dones["time_out"])

        # Increase push if robot rarely falls (good performance)
        if base_contacts < timeouts * 2:
            curr_setting_x = term_cfg.params["velocity_range"]["x"][1]
            curr_setting_x = float(np.clip(curr_setting_x * 1.5, 0.0, max_velocity[0]))
            term_cfg.params["velocity_range"]["x"] = (-curr_setting_x, curr_setting_x)

            curr_setting_y = term_cfg.params["velocity_range"]["y"][1]
            curr_setting_y = float(np.clip(curr_setting_y * 1.5, 0.0, max_velocity[1]))
            term_cfg.params["velocity_range"]["y"] = (-curr_setting_y, curr_setting_y)
            env.event_manager.set_term_cfg("push_robot", term_cfg)
            curr_setting = curr_setting_x

        # Decrease push if robot falls too often (struggling)
        if base_contacts > timeouts / 2:
            curr_setting_x = term_cfg.params["velocity_range"]["x"][1]
            curr_setting_x = float(np.clip(curr_setting_x - 0.2, 0.0, max_velocity[0]))
            term_cfg.params["velocity_range"]["x"] = (-curr_setting_x, curr_setting_x)

            curr_setting_y = term_cfg.params["velocity_range"]["y"][1]
            curr_setting_y = float(np.clip(curr_setting_y - 0.2, 0.0, max_velocity[1]))
            term_cfg.params["velocity_range"]["y"] = (-curr_setting_y, curr_setting_y)
            env.event_manager.set_term_cfg("push_robot", term_cfg)
            curr_setting = curr_setting_x

    return curr_setting


def modify_command_velocity(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    max_velocity: Sequence[float],
    interval: int,
    starting_step: float = 0.0,
) -> float:
    """Curriculum that progressively increases the command velocity range.

    When the robot tracks velocity commands well (high reward), the velocity range expands.
    This prevents early training instability from large velocity commands.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the velocity-tracking reward term.
        max_velocity: Maximum [neg_x, pos_x] velocity limits (e.g., [-2.0, 2.0]).
        interval: The number of steps between curriculum updates.
        starting_step: The step count at which to begin applying the curriculum.

    Returns:
        The current maximum forward velocity command.
    """
    command_cfg = env.command_manager.get_term("base_velocity").cfg
    curr_lin_vel_x = command_cfg.ranges.lin_vel_x

    if env.common_step_counter < starting_step:
        return curr_lin_vel_x[1]

    if env.common_step_counter % interval == 0:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        rew = env.reward_manager._episode_sums[term_name][env_ids]

        # Expand velocity range when robot tracks current commands well (>80% of max reward)
        if torch.mean(rew) / env.max_episode_length > 0.8 * term_cfg.weight * env.step_dt:
            curr_lin_vel_x = (
                float(np.clip(curr_lin_vel_x[0] - 0.5, max_velocity[0], 0.0)),
                float(np.clip(curr_lin_vel_x[1] + 0.5, 0.0, max_velocity[1])),
            )
            command_cfg.ranges.lin_vel_x = curr_lin_vel_x

    return curr_lin_vel_x[1]

# --- Paste the function directly here so we ignore the curriculums.py file entirely ---
def progressive_friction_randomization(
    env,
    env_ids: Sequence[int],
    term_name: str,
    min_friction: float,
    max_friction: float,
    interval: int,
    starting_step: int = 0,
) -> float:
    try:
        term_cfg = env.event_manager.get_term_cfg(term_name)
    except Exception:
        return 0.0

    curr_lower = term_cfg.params["static_friction_range"][0]
    curr_upper = term_cfg.params["static_friction_range"][1]

    if env.common_step_counter < starting_step:
        return curr_lower

    if env.common_step_counter % interval == 0:
        new_lower = max(min_friction, curr_lower - 0.05)
        new_upper = min(max_friction, curr_upper + 0.05)
        term_cfg.params["static_friction_range"] = (new_lower, new_upper)
        term_cfg.params["dynamic_friction_range"] = (new_lower, new_upper)
        env.event_manager.set_term_cfg(term_name, term_cfg)

    return term_cfg.params["static_friction_range"][0]