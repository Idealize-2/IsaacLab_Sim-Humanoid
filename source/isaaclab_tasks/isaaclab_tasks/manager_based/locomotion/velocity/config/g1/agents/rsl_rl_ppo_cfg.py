# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]


# ==============================================================================
# Curriculum Variant PPO Runners
# ==============================================================================


@configclass
class G1RewardCurriculumPPORunnerCfg(G1RoughPPORunnerCfg):
    """PPO runner for the G1 reward-weight curriculum environment.

    Trains longer than the base rough env since the curriculum gradually
    introduces harder objectives over 5000+ steps.
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 4000
        self.experiment_name = "g1_reward_curriculum"


@configclass
class G1PushCurriculumPPORunnerCfg(G1RoughPPORunnerCfg):
    """PPO runner for the G1 adaptive push-force curriculum environment.

    Uses a longer training horizon so the push force has time to scale up
    from the warm-up phase (step 1000) to the maximum value.
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 4000
        self.experiment_name = "g1_push_curriculum"


@configclass
class G1VelocityCurriculumPPORunnerCfg(G1RoughPPORunnerCfg):
    """PPO runner for the G1 progressive velocity-command curriculum environment.

    Runs for the same number of iterations as the rough env; the curriculum
    ensures early training is stable and the task becomes harder over time.
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3500
        self.experiment_name = "g1_velocity_curriculum"


@configclass
class G1FullCurriculumPPORunnerCfg(G1RoughPPORunnerCfg):
    """PPO runner for the G1 full multi-stage curriculum environment.

    Requires the most iterations since the full curriculum combines terrain,
    push, velocity, and reward-weight schedules that span beyond step 5000.
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 6000
        self.experiment_name = "g1_full_curriculum"
        # Slightly larger networks for the more complex full-curriculum task
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
