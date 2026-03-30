# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DigitRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "digit_rough"
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
        entropy_coef=0.01,
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
class DigitFlatPPORunnerCfg(DigitRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 2000
        self.experiment_name = "digit_flat"

        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]

# ==============================================================================
# Curriculum Variant PPO Runners
# ==============================================================================


@configclass
class Digit1RewardCurriculumPPORunnerCfg(DigitRoughPPORunnerCfg):
    """PPO runner for the digit reward-weight curriculum environment.

    Trains longer than the base rough env since the curriculum gradually
    introduces harder objectives over 5000+ steps.
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 4000
        self.experiment_name = "digit_reward_curriculum"


@configclass
class DigitPushCurriculumPPORunnerCfg(DigitRoughPPORunnerCfg):
    """PPO runner for the digit adaptive push-force curriculum environment.

    Uses a longer training horizon so the push force has time to scale up
    from the warm-up phase (step 1000) to the maximum value.
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 4000
        self.experiment_name = "digit_push_curriculum"


@configclass
class DigitVelocityCurriculumPPORunnerCfg(DigitRoughPPORunnerCfg):
    """PPO runner for the digit progressive velocity-command curriculum environment.

    Runs for the same number of iterations as the rough env; the curriculum
    ensures early training is stable and the task becomes harder over time.
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3500
        self.experiment_name = "digit_velocity_curriculum"


@configclass
class DigitFullCurriculumPPORunnerCfg(DigitRoughPPORunnerCfg):
    """PPO runner for the digit full multi-stage curriculum environment.

    Requires the most iterations since the full curriculum combines terrain,
    push, velocity, and reward-weight schedules that span beyond step 5000.
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 6000
        self.experiment_name = "digit_full_curriculum"
        # Slightly larger networks for the more complex full-curriculum task
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

