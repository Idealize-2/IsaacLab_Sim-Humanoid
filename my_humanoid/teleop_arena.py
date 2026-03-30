# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import traceback
from isaaclab.app import AppLauncher

# 1. Setup Custom Arguments
parser = argparse.ArgumentParser(description="Teleop your trained robots in a custom USD arena.")
parser.add_argument("--task", type=str, required=True, help="Name of the task (e.g. Isaac-Velocity-Rough-G1-FullCurriculum-v0)")
parser.add_argument("--usd", type=str, required=True, help="Path to your custom arena USD file")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to your trained model .pt file")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robots to spawn")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import carb
import omni.appwindow

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
from isaaclab.terrains import TerrainImporterCfg

class TeleopController:
    """Handles keyboard inputs to override robot velocity commands."""
    def __init__(self, device):
        self.device = device
        self.command = torch.zeros(3, device=device) # [x_vel, y_vel, yaw_vel]
        
        # Sprint Settings (Change these to match your curriculum limits!)
        self.max_lin_vel = 4.0 
        self.max_ang_vel = 2.0
        
        self.appwindow = omni.appwindow.get_default_app_window()
        self.input_interface = carb.input.acquire_input_interface()
        self.keyboard = self.appwindow.get_keyboard()
        self.sub_input = self.input_interface.subscribe_to_keyboard_events(self.keyboard, self._keyboard_event_handler)
        
        print("\n" + "="*50)
        print("[INFO] ARENA TELEOP CONTROLS ENGAGED")
        print("  UP/W       - Sprint forward")
        print("  DOWN/S     - Stop")
        print("  LEFT/A     - Turn left")
        print("  RIGHT/D    - Turn right")
        print("  ESC        - Quit")
        print("="*50 + "\n")

    def _keyboard_event_handler(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in ["UP", "W"]:
                self.command[0] = self.max_lin_vel
            elif event.input.name in ["DOWN", "S"]:
                self.command[0] = 0.0
            elif event.input.name in ["LEFT", "A"]:
                self.command[2] = self.max_ang_vel
            elif event.input.name in ["RIGHT", "D"]:
                self.command[2] = -self.max_ang_vel
            elif event.input.name == "ESCAPE":
                simulation_app.close()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["UP", "DOWN", "W", "S"]:
                self.command[0] = 0.0
            elif event.input.name in ["LEFT", "RIGHT", "A", "D"]:
                self.command[2] = 0.0
        return True

def main():
    # 2. Parse Environment and Agent Configs
    env_cfg = parse_env_cfg(args_cli.task)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # 3. OVERRIDE TERRAIN
    print(f"[INFO] Injecting custom arena: {args_cli.usd}")
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=args_cli.usd,
    )
    
    # SENSOR HACK PART 1: Delete the crashy laser scanner from the environment entirely
    env_cfg.scene.height_scanner = None
    if "height_scan" in env_cfg.observations.policy.__dict__:
        env_cfg.observations.policy.height_scan = None
        
    env_cfg.curriculum = None
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.2)
    
    # 4. Create the Environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # SENSOR HACK PART 2: Trick PyTorch into building a 310-input brain even though we only have 123 sensors active
    env.num_obs = 310 

    # 5. Load Your Specific Brain
    resume_path = os.path.abspath(args_cli.checkpoint)
    print(f"\n[INFO] Loading trained policy from: {resume_path}\n")
    
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 6. Initialize Controller
    teleop = TeleopController(env.unwrapped.device)

    # 7. The Play Loop
    print("[INFO] Resetting environment and starting simulation...")
    obs, _ = env.reset() 
    
    with torch.inference_mode():
        while simulation_app.is_running():
            # INJECT KEYBOARD COMMAND
            obs[:, 9:12] = teleop.command
            
            # SENSOR HACK PART 3: Pad the 123 active sensors with 187 zeros to simulate a perfectly flat floor
            obs_padded = torch.zeros((obs.shape[0], 310), dtype=obs.dtype, device=obs.device)
            obs_padded[:, :123] = obs # Copy the real joint data in
            
            # Get AI action using the padded tensor
            actions = policy(obs_padded)
            obs, _, _, _ = env.step(actions)

    env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "!"*50)
        print("CRASH DETECTED! HERE IS THE ERROR:")
        traceback.print_exc()
        print("!"*50 + "\n")
    finally:
        simulation_app.close()