# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg 

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CurriculumCfg

from .rough_env_cfg import DigitRoughEnvCfg

# --- Safe Friction Wrapper ---
_friction_randomizer = None

def safe_randomize_friction(env, env_ids, asset_cfg, static_friction_range, dynamic_friction_range, restitution_range, num_buckets):
    global _friction_randomizer
    if _friction_randomizer is None:
        from isaaclab.managers import EventTermCfg
        dummy_cfg = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": asset_cfg,
                "static_friction_range": static_friction_range,
                "dynamic_friction_range": dynamic_friction_range,
                "restitution_range": restitution_range,
                "num_buckets": num_buckets
            }
        )
        _friction_randomizer = mdp.randomize_rigid_body_material(dummy_cfg, env)
    _friction_randomizer(env, env_ids)

# ==============================================================================
# Curriculum Term Configurations
# ==============================================================================

@configclass
class DigitFullCurriculumCfg(CurriculumCfg):
    """Full multi-stage curriculum (terrain + velocity) optimized for sprinting."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    command_velocity = CurrTerm(
        func=mdp.modify_command_velocity,
        params={
            "term_name": "track_lin_vel_xy_exp",
            "max_velocity": [-1.0, 5.0], # <--- Sprint up to 5 m/s!
            "interval": 100,
            "starting_step": 500,
        },
    )

# ==============================================================================
# Environment Configurations
# ==============================================================================

@configclass
class DigitFullCurriculumEnvCfg(DigitRoughEnvCfg):
    
    curriculum: DigitFullCurriculumCfg = DigitFullCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        
        # --- 1. COMMANDS ---
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0) 
        self.commands.base_velocity.ranges.ang_vel_z = (-0.1, 0.1) 
        
        # --- 2. HACK THE REWARDS FOR MAXIMUM SPEED ---
        self.rewards.track_lin_vel_xy_exp.weight = 5.0 
        
        # Turn OFF all safety and energy penalties
        self.rewards.dof_torques_l2.weight = 0.0       
        self.rewards.dof_acc_l2.weight = 0.0           
        self.rewards.action_rate_l2.weight = 0.0       
        self.rewards.flat_orientation_l2.weight = 0.0  
        
        # Digit Specific joint deviation penalties turned off (if they exist)
        if hasattr(self.rewards, "joint_deviation_hip"):
            self.rewards.joint_deviation_hip.weight = 0.0  
        if hasattr(self.rewards, "joint_deviation_torso"):
            self.rewards.joint_deviation_torso.weight = 0.0
        if hasattr(self.rewards, "joint_deviation_arms"):
            self.rewards.joint_deviation_arms.weight = 0.0

        # --- 3. EVENTS ---
        # Target Digit's toe links for friction randomization
        # self.events.randomize_foot_friction = EventTerm(
        #     func=safe_randomize_friction,
        #     mode="reset",
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=".*toe.*"), # Standard Digit contact point
        #         "static_friction_range": (0.3, 1.5),
        #         "dynamic_friction_range": (0.3, 1.5),
        #         "restitution_range": (0.0, 0.0),
        #         "num_buckets": 64,
        #     },
        # )

@configclass
class DigitFullCurriculumEnvCfg_PLAY(DigitFullCurriculumEnvCfg):
    """Play/evaluation variant of DigitFullCurriculumEnvCfg."""

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