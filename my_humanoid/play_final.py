# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import pathlib
import traceback
from isaaclab.app import AppLauncher

# 1. Setup Arguments
parser = argparse.ArgumentParser(description="Interactive humanoid control with custom USD scene.")
parser.add_argument("--task", type=str, required=True, help="Task name (e.g., Isaac-Velocity-Rough-G1-FullCurriculum-v0)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
parser.add_argument("--usd", type=str, required=True, help="Path to your custom USD arena")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import carb
import omni.appwindow

from isaaclab.terrains import TerrainImporterCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

class InteractiveController:
    """Handles keyboard inputs to override robot velocity commands with smooth acceleration."""
    def __init__(self, device):
        self.device = device
        
        # We now have a TARGET command (where we want to go) and a CURRENT command (where we actually are)
        self.target_command = torch.zeros(3, device=device) # [x_vel, y_vel, yaw_vel]
        self.current_command = torch.zeros(3, device=device) 
        
        # Speeds
        self.max_lin_vel = 1.5  # Slightly lowered for better stability
        self.max_ang_vel = 1.0
        
        self.appwindow = omni.appwindow.get_default_app_window()
        self.input_interface = carb.input.acquire_input_interface()
        self.keyboard = self.appwindow.get_keyboard()
        self.sub_input = self.input_interface.subscribe_to_keyboard_events(self.keyboard, self._keyboard_event_handler)
        
        print("\n" + "="*50)
        print("[INFO] SMOOTH KEYBOARD CONTROLS ENGAGED")
        print("  UP/W       - Sprint forward")
        print("  DOWN/S     - Stop")
        print("  LEFT/A     - Turn left")
        print("  RIGHT/D    - Turn right")
        print("  ESC        - Quit")
        print("="*50 + "\n")

    def _keyboard_event_handler(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in ["UP", "W"]:
                self.target_command[0] = self.max_lin_vel
            elif event.input.name in ["DOWN", "S"]:
                self.target_command[0] = 0.0
            elif event.input.name in ["LEFT", "A"]:
                self.target_command[2] = self.max_ang_vel
            elif event.input.name in ["RIGHT", "D"]:
                self.target_command[2] = -self.max_ang_vel
            elif event.input.name == "ESCAPE":
                simulation_app.close()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["UP", "DOWN", "W", "S"]:
                self.target_command[0] = 0.0
            elif event.input.name in ["LEFT", "RIGHT", "A", "D"]:
                self.target_command[2] = 0.0
        return True
        
    def get_smoothed_command(self):
        # This acts like a gas pedal! It blends 95% of the old speed with 5% of the new speed every frame.
        self.current_command = 0.95 * self.current_command + 0.05 * self.target_command
        return self.current_command

def main():
    print(f"[INFO] Loading task: {args_cli.task}")
    env_cfg = parse_env_cfg(args_cli.task)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # Force the path into a strict Omniverse 'file:///' URI
    raw_path = pathlib.Path(args_cli.usd).expanduser().resolve()
    full_usd_path = raw_path.as_uri() 
    
    print(f"[INFO] Injecting custom arena from: {full_usd_path}")
    
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=full_usd_path,
    )
    
    # RESTORE THE REAL HEIGHT SCANNER!
    # Because your USD file is fixed, we just point the lasers directly at the new ground folder.
    if hasattr(env_cfg.scene, "height_scanner") and env_cfg.scene.height_scanner is not None:
        env_cfg.scene.height_scanner.mesh_prim_paths = ["/World/ground"]
        
    env_cfg.curriculum = None
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.robot.init_state.pos = (-5.5, 0.0, 1.2) # Drop from sky
    
    # Configure command ranges
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 4.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-2.0, 3.0)

    print(f"[INFO] Creating environment with {env_cfg.scene.num_envs} robot(s)...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    resume_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading model checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    controller = InteractiveController(env.unwrapped.device)

    print("[INFO] Starting simulation. Use controls to move the robot.")
    obs, _ = env.reset() 
    
    with torch.inference_mode():
        while simulation_app.is_running():
            
            # # 1. Get the smoothed gas pedal command
            # raw_command = controller.get_smoothed_command()
            
            # # 2. SCALE THE COMMANDS for the Neural Network!
            # scaled_command = raw_command.clone()
            # scaled_command[0] = raw_command[0] * 2.0   # Standard linear scale
            # scaled_command[1] = raw_command[1] * 2.0
            # scaled_command[2] = raw_command[2] * 0.25  # Standard angular scale
            
            # # 3. Inject the properly scaled commands
            # obs[:, 9:12] = scaled_command

            obs[:, 9:12] = controller.get_smoothed_command()
            
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "!"*50)
        print("CRASH DETECTED! ERROR:")
        traceback.print_exc()
        print("!"*50 + "\n")
    finally:
        simulation_app.close()