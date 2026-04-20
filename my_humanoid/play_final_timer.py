# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import pathlib
import traceback
import time  
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
import omni.ui as ui 

from isaaclab.terrains import TerrainImporterCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

class TimerUI:
    """Creates a floating GUI window inside Isaac Sim to display lap times."""
    def __init__(self):
        self.window = ui.Window("⏱️ Challenge Timer", width=300, height=180, 
                                flags=ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_RESIZE)
        
        with self.window.frame:
            with ui.VStack(spacing=10, style={"margin": 15}):
                ui.Label("CURRENT LAP:", style={"color": 0xFFAAAAAA, "font_size": 14})
                self.current_label = ui.Label("0.00s", 
                                              style={"color": 0xFF00FF00, "font_size": 48, "alignment": ui.Alignment.CENTER}) 
                
                ui.Spacer(height=5)
                
                ui.Label("BEST TIME:", style={"color": 0xFFAAAAAA, "font_size": 14})
                self.best_label = ui.Label("--.--s", 
                                           style={"color": 0xFFFFBB00, "font_size": 24, "alignment": ui.Alignment.CENTER}) 

    def update_display(self, current_time, best_time, is_running):
        self.current_label.text = f"{current_time:.2f}s"
        
        if is_running:
            self.current_label.style = {"color": 0xFF00FF00, "font_size": 48, "alignment": ui.Alignment.CENTER} 
        else:
            self.current_label.style = {"color": 0xFFFFFFFF, "font_size": 48, "alignment": ui.Alignment.CENTER} 
            
        if best_time < 999.0:
            self.best_label.text = f"{best_time:.2f}s"


class InteractiveController:
    """Handles keyboard inputs to override robot velocity commands with smooth acceleration."""
    def __init__(self, device):
        self.device = device
        self.target_command = torch.zeros(3, device=device) 
        self.current_command = torch.zeros(3, device=device) 
        
        self.reset_flag = False 
        
        self.max_lin_vel = 4.0  
        self.max_ang_vel = 1.5
        
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
        print("  R          - Reset Robot & Timer") 
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
            elif event.input.name == "R":
                self.reset_flag = True 
            elif event.input.name == "ESCAPE":
                simulation_app.close()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["UP", "DOWN", "W", "S"]:
                self.target_command[0] = 0.0
            elif event.input.name in ["LEFT", "RIGHT", "A", "D"]:
                self.target_command[2] = 0.0
        return True
        
    def get_smoothed_command(self):
        self.current_command = 0.95 * self.current_command + 0.05 * self.target_command
        return self.current_command

def main():
    print(f"[INFO] Loading task: {args_cli.task}")
    env_cfg = parse_env_cfg(args_cli.task)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    raw_path = pathlib.Path(args_cli.usd).expanduser().resolve()
    full_usd_path = raw_path.as_uri() 
    
    print(f"[INFO] Injecting custom arena from: {full_usd_path}")
    
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=full_usd_path,
    )
    
    if hasattr(env_cfg.scene, "height_scanner") and env_cfg.scene.height_scanner is not None:
        env_cfg.scene.height_scanner.mesh_prim_paths = ["/World/ground"]
        
    env_cfg.curriculum = None
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.robot.init_state.pos = (-6, 0.0, 1.2) # Drop from sky
    
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 4.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)

    print(f"[INFO] Creating environment with {env_cfg.scene.num_envs} robot(s)...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    resume_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading model checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    controller = InteractiveController(env.unwrapped.device)
    gui_timer = TimerUI() 

    # --- TIMER SETUP VARIABLES ---
    FINISH_LINE_X = 6.0
    timer_running = False
    timer_finished = False
    start_time = 0.0
    best_time = 999.9
    display_time = 0.0 
    # -----------------------------

    print("[INFO] Starting simulation. Use controls to move the robot.")
    obs, _ = env.reset() 
    
    with torch.inference_mode():
        while simulation_app.is_running():
            
            # --- MANUAL RESET CATCHER ---
            if controller.reset_flag:
                obs, _ = env.reset()
                controller.reset_flag = False
                continue 
            
            # --- AUTOMATIC TIMER LOGIC ---
            root_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
            current_x = root_pos[0].item()
            current_z = root_pos[2].item()

            # --- 1.5 AUTO-RESPAWN (THE NEW FIX) ---
            # If the robot's root drops below -1.0, it fell off the map
            if current_z < -1.0:
                print("\n[WARNING] ⚠️ Robot fell off the map! Auto-respawning...")
                controller.reset_flag = True
                continue # Skip the rest of this frame so it cleanly resets on the next tick!
            # ---------------------------------------
            
            # 2. Detect Respawn 
            if current_z > 1.1:
                if timer_running or timer_finished:
                    print("\n[TIMER] 🔄 Robot respawned. Resetting clock...")
                timer_running = False
                timer_finished = False
                display_time = 0.0
                
            # 3. Detect Landing 
            elif not timer_running and not timer_finished and current_z < 0.95:
                timer_running = True
                start_time = time.time()
                print("\n" + "="*40)
                print("[TIMER] 🟢 GO! Robot touched the ground!")
                print("="*40)
                
            # 4. Detect Finish Line 
            elif timer_running and current_x > FINISH_LINE_X:
                elapsed_time = time.time() - start_time
                display_time = elapsed_time 
                
                print("\n" + "="*40)
                print(f"[TIMER] 🏁 FINISH LINE CROSSED!")
                print(f"[TIMER] Lap Time: {elapsed_time:.2f} seconds")
                
                if elapsed_time < best_time:
                    best_time = elapsed_time
                    print(f"[TIMER] 🏆 NEW BEST TIME: {best_time:.2f}s!")
                print("="*40 + "\n")
                
                timer_running = False
                timer_finished = True
                
            # 5. Calculate running time 
            if timer_running:
                display_time = time.time() - start_time
                
            # 6. Send current status to the Omniverse GUI
            gui_timer.update_display(display_time, best_time, timer_running)
            
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