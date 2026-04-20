# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 Humanoid Curriculum Environments.

This module defines four curriculum variants for the G1 humanoid robot velocity-tracking task:

1. **RewardCurriculum**  – Progressively increases reward/penalty weights after the robot
   has mastered basic locomotion (step-based scheduling).

2. **PushCurriculum**    – Starts with a small external push force and adaptively scales it
   based on how often the robot falls vs. completes episodes.

3. **VelocityCurriculum** – Begins with a narrow velocity-command range and expands it once
   the robot reliably tracks lower-speed commands.

4. **FullCurriculum**    – Combines all three curricula above for the most comprehensive
   multi-stage training schedule.

Reference:
    https://isaac-sim.github.io/IsaacLab/main/source/how-to/curriculums.html
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CurriculumCfg

from isaaclab.managers import SceneEntityCfg 

from .rough_env_cfg import G1RoughEnvCfg

# ==============================================================================
# Curriculum Term Configurations
# ==============================================================================


@configclass
class G1RewardCurriculumCfg(CurriculumCfg):
    """Curriculum: terrain difficulty + step-based reward weight scheduling.

    Stage 1 (step 0 → 1500):   baseline training with light penalties.
    Stage 2 (step 1500):        orientation penalty increases to -2.0.
    Stage 3 (step 3000):        feet air-time reward increases to 0.5.
    Stage 4 (step 5000):        hip-deviation penalty increases to -0.2.
    """

    # Terrain difficulty always active
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # Step 1500: strengthen flat-orientation penalty
    flat_orientation_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "flat_orientation_l2", "weight": -2.0, "num_steps": 1500},
    )
    # Step 3000: boost feet air-time reward
    feet_air_time_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "feet_air_time", "weight": 0.5, "num_steps": 3000},
    )
    # Step 5000: increase hip-yaw / hip-roll deviation penalty
    joint_deviation_hip_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_deviation_hip", "weight": -0.2, "num_steps": 5000},
    )


@configclass
class G1PushCurriculumCfg(CurriculumCfg):
    """Curriculum: terrain difficulty + adaptive push-force escalation.

    Push velocity starts at ±0.1 m/s and is scaled up every 200 steps
    (max ±1.5 m/s) when fall rate is low, or scaled down when falls are frequent.
    The curriculum activates after step 1000 to give the robot a warm-up phase.
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # push_force = CurrTerm(
    #     func=mdp.modify_push_force,
    #     params={
    #         "term_name": "push_robot",
    #         "max_velocity": [1.5, 1.5],   # max [x, y] push m/s
    #         "interval": 200,              # re-evaluate every 200 steps
    #         "starting_step": 1000,        # warm-up: no scaling before step 1000
    #     },
    # )


@configclass
class G1VelocityCurriculumCfg(CurriculumCfg):
    """Curriculum: terrain difficulty + progressive velocity-command expansion.

    Starts with lin_vel_x ∈ [0, 0.5] m/s.  Every 100 steps the range expands by
    ±0.5 m/s (up to ±2.0 m/s) whenever the robot achieves >80 % of the max
    tracking reward, ensuring the task stays challenging but reachable.
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    command_velocity = CurrTerm(
        func=mdp.modify_command_velocity,
        params={
            "term_name": "track_lin_vel_xy_exp",
            "max_velocity": [-2.0, 2.0],  # [min_x, max_x]
            "interval": 100,              # check every 100 steps
            "starting_step": 500,         # warm-up: no expansion before step 500
        },
    )

    friction_curriculum = CurrTerm(
        func=mdp.progressive_friction_randomization,
        params={
            "term_name": "randomize_foot_friction", 
            "min_friction": 0.3,   
            "max_friction": 1.3,   
            "interval": 500,       
            "starting_step": 1000, 
        }
    )


# @configclass
# class G1FullCurriculumCfg(CurriculumCfg):
#     """Full multi-stage curriculum (terrain + push + velocity + reward weights).

#     Combines all curriculum terms:
#     - Terrain difficulty scales with robot performance.
#     - Push force grows adaptively (warm-up to step 1000).
#     - Velocity range expands progressively (warm-up to step 500).
#     - Orientation penalty ramps up at step 2000.
#     - Hip-deviation penalty ramps up at step 5000.
#     """

#     terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

#     push_force = CurrTerm(
#         func=mdp.modify_push_force,
#         params={
#             "term_name": "push_robot",
#             "max_velocity": [2.0, 2.0],
#             "interval": 200,
#             "starting_step": 1000,
#         },
#     )

#     command_velocity = CurrTerm(
#         func=mdp.modify_command_velocity,
#         params={
#             "term_name": "track_lin_vel_xy_exp",
#             "max_velocity": [-2.0, 2.0],
#             "interval": 100,
#             "starting_step": 500,
#         },
#     )

#     flat_orientation_weight = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name": "flat_orientation_l2", "weight": -2.0, "num_steps": 2000},
#     )

#     joint_deviation_hip_weight = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name": "joint_deviation_hip", "weight": -0.2, "num_steps": 5000},
#     )

@configclass
class G1FullCurriculumCfg(CurriculumCfg):
    """Full multi-stage curriculum (terrain + push + velocity)."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    push_force = CurrTerm(
        func=mdp.modify_push_force,
        params={
            "term_name": "push_robot",
            "max_velocity": [2.0, 2.0],
            "interval": 200,
            "starting_step": 1000,
        },
    )

    command_velocity = CurrTerm(
        func=mdp.modify_command_velocity,
        params={
            "term_name": "track_lin_vel_xy_exp",
            "max_velocity": [-1.0, 5.0], # <--- CHANGED: Let the robot learn to sprint up to 5 m/s!
            "interval": 100,
            "starting_step": 500,
        },
    )


# ==============================================================================
# Environment Configurations
# ==============================================================================


@configclass
class G1RewardCurriculumEnvCfg(G1RoughEnvCfg):
    """G1 rough-terrain env with step-based reward-weight curriculum.

    Inherits all G1 rough settings.  Curriculum stages:
      • Steps 0-1500:  orientation penalty = 0  (robot learns to walk first)
      • Steps 1500+:   orientation penalty = -2.0
      • Steps 3000+:   feet air-time reward = 0.5
      • Steps 5000+:   hip-deviation penalty = -0.2
    """

    curriculum: G1RewardCurriculumCfg = G1RewardCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Curriculum will gradually enable/increase these terms
        self.rewards.flat_orientation_l2.weight = 0.0    # starts at zero
        self.rewards.joint_deviation_hip.weight = -0.05  # starts light
        self.rewards.feet_air_time.weight = 0.1          # starts light


@configclass
class G1RewardCurriculumEnvCfg_PLAY(G1RewardCurriculumEnvCfg):
    """Play/evaluation variant of G1RewardCurriculumEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ------------------------------------------------------------------------------


@configclass
class G1PushCurriculumEnvCfg(G1RoughEnvCfg):
    """G1 rough-terrain env with adaptive push-force curriculum.

    Push robot starts disabled (force ≈ 0) and is gradually increased
    by the curriculum based on fall rate.  Curriculum activates at step 1000.
    Max push velocity: ±1.5 m/s in x and y.
    """

    curriculum: G1PushCurriculumCfg = G1PushCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Re-enable push_robot with a small initial force (G1RoughEnvCfg sets it to None)
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
        )


@configclass
class G1PushCurriculumEnvCfg_PLAY(G1PushCurriculumEnvCfg):
    """Play/evaluation variant of G1PushCurriculumEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ------------------------------------------------------------------------------


@configclass
class G1VelocityCurriculumEnvCfg(G1RoughEnvCfg):
    """G1 rough-terrain env with progressive velocity-command curriculum.

    Starts with a narrow velocity command range:
      lin_vel_x ∈ [0.0, 0.5] m/s, lin_vel_y ∈ [-0.2, 0.2] m/s, ang_vel_z ∈ [-0.5, 0.5]

    The curriculum expands lin_vel_x by ±0.5 m/s every 100 steps (once the robot
    achieves >80 % tracking reward), up to ±2.0 m/s.
    """

    curriculum: G1VelocityCurriculumCfg = G1VelocityCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Narrow initial velocity range; curriculum will expand it
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class G1VelocityCurriculumEnvCfg_PLAY(G1VelocityCurriculumEnvCfg):
    """Play/evaluation variant of G1VelocityCurriculumEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ------------------------------------------------------------------------------


# @configclass
# class G1FullCurriculumEnvCfg(G1RoughEnvCfg):
#     """G1 rough-terrain env with the full multi-stage curriculum.

#     Combines all four curriculum strategies:
#     1. **Terrain levels** – harder terrain as the robot walks further.
#     2. **Adaptive push** – push force scales from ±0.1 → ±2.0 m/s (starts at step 1000).
#     3. **Velocity expansion** – lin_vel_x expands from [0, 0.5] → [-2.0, 2.0] m/s (starts at step 500).
#     4. **Reward scheduling** – orientation penalty added at step 2000; hip-deviation at step 5000.
#     """

#     curriculum: G1FullCurriculumCfg = G1FullCurriculumCfg()

#     def __post_init__(self):
#         super().__post_init__()
#         # Narrow initial velocity range - curriculum expands it progressively
#         self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
#         self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
#         self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
#         # Start with zero orientation penalty - curriculum will enable at step 2000
#         self.rewards.flat_orientation_l2.weight = 0.0
#         # Start with lighter hip-deviation penalty - curriculum increases at step 5000
#         self.rewards.joint_deviation_hip.weight = -0.05
#         # Re-enable push_robot with small initial force (curriculum scales it up)
#         self.events.push_robot = EventTerm(
#             func=mdp.push_by_setting_velocity,
#             mode="interval",
#             interval_range_s=(10.0, 15.0),
#             params={"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
#         )

#         # Start with normal friction. The curriculum will widen this range later!
#         self.events.randomize_foot_friction = EventTerm(
#             func=mdp.randomize_rigid_body_material,
#             mode="reset",
#             params={
#                 "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot.*"), # Make sure this matches your G1 foot link names!
#                 "static_friction_range": (0.8, 0.8),
#                 "dynamic_friction_range": (0.8, 0.8),
#                 "restitution_range": (0.0, 0.0),
#                 "num_buckets": 64,
#             },
#         )

@configclass
class G1FullCurriculumEnvCfg(G1RoughEnvCfg):
    
    curriculum: G1FullCurriculumCfg = G1FullCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        
        # Start with a moderate jog, the curriculum will expand it up to 5.0 m/s
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2) 
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5) 
        
        # Massive incentive to track the high-speed push_force commands
        self.rewards.track_lin_vel_xy_exp.weight = 5.0 
        self.rewards.track_ang_vel_z_exp.weight = 2.5
        
        # Turn OFF all safety and energy penalties (set to 0.0)
        self.rewards.dof_torques_l2.weight = 0.0       # Use maximum motor power
        self.rewards.dof_acc_l2.weight = 0.0           # Allow violent leg swings
        self.rewards.action_rate_l2.weight = 0.0       # Allow rapid twitching
        self.rewards.flat_orientation_l2.weight = 0.0  # Let the torso lean like a sprinter
        self.rewards.joint_deviation_hip.weight = 0.0  # Ignore ugly posture
        self.rewards.joint_deviation_torso.weight = 0.0
        self.rewards.joint_deviation_arms.weight = 0.0
        
        # self.events.push_robot = EventTerm(
        #     func=mdp.push_by_setting_velocity,
        #     mode="interval",
        #     interval_range_s=(10.0, 15.0),
        #     params={"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
        # )

        # self.events.randomize_foot_friction = EventTerm(
        #     func=mdp.randomize_rigid_body_material,
        #     mode="reset",
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot.*"), # Make sure this matches your G1 foot link names!
        #         "static_friction_range": (0.8, 0.8),
        #         "dynamic_friction_range": (0.8, 0.8),
        #         "restitution_range": (0.0, 0.0),
        #         "num_buckets": 64,
        #     },
        # )


@configclass
class G1FullCurriculumEnvCfg_PLAY(G1FullCurriculumEnvCfg):
    """Play/evaluation variant of G1FullCurriculumEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.events.randomize_foot_friction = None
