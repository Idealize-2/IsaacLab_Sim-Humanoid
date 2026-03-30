# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Interactive G1 Humanoid Control with Scene Selection

This script allows you to:
- Choose between different scenes (terrain, warehouse, empty)
- Control the G1 robot with keyboard or gamepad
- Use a trained policy checkpoint

Usage:
    # With keyboard control in warehouse
    ./isaaclab.sh -p my_humanoid/play_interactive.py --checkpoint /path/to/model.pt --scene warehouse

    # With gamepad control on rough terrain
    ./isaaclab.sh -p my_humanoid/play_interactive.py --checkpoint /path/to/model.pt --scene terrain --use_gamepad
"""

import argparse


from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Interactive G1 humanoid control with scene selection.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (JIT .pt file)")
parser.add_argument(
    "--scene",
    type=str,
    default="terrain",
    choices=["terrain", "warehouse", "empty"],
    help="Scene type: 'terrain' (rough terrain), 'warehouse' (USD environment), 'empty' (flat plane)",
)
parser.add_argument("--use_gamepad", action="store_true", help="Use gamepad instead of keyboard")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")

# Append AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import os

import torch
import carb
import omni
import omni.appwindow
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_tasks.manager_based.locomotion.velocity.config.digit.flat_env_cfg import DigitFlatEnvCfg_PLAY


class G1InteractivePolicy:
    """G1 Humanoid with interactive control."""

    def __init__(self, policy_model, device, use_gamepad=False):
        """
        Initialize G1 interactive policy.

        Args:
            policy_model: The loaded JIT policy model
            device: Device to run on (cuda/cpu)
            use_gamepad: Whether to use gamepad or keyboard
        """
        self.policy = policy_model
        self.device = device
        self.use_gamepad = use_gamepad
        self.base_command = torch.zeros(3, device=device)  # [lin_vel_x, lin_vel_y, ang_vel_z]


        # Control parameters
        
        #self.max_lin_vel = 0.8
        #self.max_ang_vel = 1.0
        #self.gamepad_deadzone = 0.15

        self.max_lin_vel = 1.2
        self.max_ang_vel = 1.5
        self.gamepad_deadzone = 0.2

        # Setup input
        self.appwindow = omni.appwindow.get_default_app_window()
        self.input_interface = carb.input.acquire_input_interface()

        if use_gamepad:
            self._setup_gamepad()
        else:
            self._setup_keyboard()

    def _setup_keyboard(self):
        """Setup keyboard input."""
        keyboard = self.appwindow.get_keyboard()
        self.sub_input = self.input_interface.subscribe_to_keyboard_events(keyboard, self._keyboard_event_handler)
        print("\n[INFO] Keyboard Controls:")
        print("  UP/W       - Move forward")
        print("  DOWN/S     - Stop")
        print("  LEFT/A     - Turn left")
        print("  RIGHT/D    - Turn right")
        print("  ESC        - Quit\n")

    def _setup_gamepad(self):
        """Setup gamepad input."""
        gamepad = self.appwindow.get_gamepad(0)
        self.sub_input = self.input_interface.subscribe_to_gamepad_events(gamepad, self._gamepad_event_handler)
        print("\n[INFO] Gamepad Controls:")
        print("  Left Stick Y  - Forward/Backward")
        print("  Right Stick X - Turn Left/Right")
        print("  Deadzone: {}\n".format(self.gamepad_deadzone))

    def _keyboard_event_handler(self, event, *args, **kwargs):
        """Handle keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in ["UP", "W"]:
                self.base_command[0] = self.max_lin_vel
            elif event.input.name in ["DOWN", "S"]:
                self.base_command[0] = 0.0
            elif event.input.name in ["LEFT", "A"]:
                self.base_command[2] = self.max_ang_vel
            elif event.input.name in ["RIGHT", "D"]:
                self.base_command[2] = -self.max_ang_vel
            elif event.input.name == "ESCAPE":
                simulation_app.close()

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["UP", "DOWN", "W", "S"]:
                self.base_command[0] = 0.0
            elif event.input.name in ["LEFT", "RIGHT", "A", "D"]:
                self.base_command[2] = 0.0

        return True

    def _gamepad_event_handler(self, event, *args, **kwargs):
        """Handle gamepad events."""
        if event.type == carb.input.GamepadEventType.AXIS:
            if event.input == carb.input.GamepadInput.LEFT_STICK_UP:
                # Forward/backward (inverted because up is negative)
                value = -event.value if abs(event.value) > self.gamepad_deadzone else 0.0
                self.base_command[0] = value * self.max_lin_vel

            elif event.input == carb.input.GamepadInput.RIGHT_STICK_RIGHT:
                # Yaw rotation
                value = event.value if abs(event.value) > self.gamepad_deadzone else 0.0
                self.base_command[2] = value * self.max_ang_vel

        return True

    def compute_action(self, obs):
        """
        Compute action from observation with command override.

        Args:
            obs: Observation dict from environment

        Returns:
            Action tensor
        """
        # Override velocity command in observation (indices 9:12 for G1)
        obs["policy"][:, 9:12] = self.base_command
        return self.policy(obs["policy"])

    def cleanup(self):
        """Cleanup input subscriptions."""
        if hasattr(self, "sub_input"):
            self.sub_input.unsubscribe()


def setup_scene(env_cfg, scene_type):
    """
    Setup the scene based on user selection.

    Args:
        env_cfg: Environment configuration
        scene_type: Type of scene ('terrain', 'warehouse', 'empty')

    Returns:
        Modified environment configuration
    """
    if scene_type == "warehouse":
        print("[INFO] Loading warehouse scene...")
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        )
        # Disable height scanner for flat environment
        env_cfg.scene.height_scanner = None
        env_cfg.observations.policy.height_scan = None

    elif scene_type == "empty":
        print("[INFO] Loading empty flat scene...")
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            terrain_generator=None,
        )
        # Disable height scanner for flat environment
        env_cfg.scene.height_scanner = None
        env_cfg.observations.policy.height_scan = None
    elif scene_type == "terrain":
        # ต้องสร้าง TerrainImporterCfg ใหม่ให้เป็นแบบ generator
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                size=(8.0, 8.0),
                border_width=20.0,
                num_rows=1,
                num_cols=1,
                sub_terrains={
                    "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                        proportion=1.0,
                        slope_range=(0.0, 20), # Up to ~22 degrees
                        platform_width=3.0,     # Flat area at the top
                        border_width=0.25,
                    ),
                },
            ),
        )
    
        # ปิด scanner ถ้าไม่ใช่ฉาก terrain เพื่อกัน crash
        if scene_type != "terrain":
            env_cfg.scene.height_scanner = None
            env_cfg.observations.policy.height_scan = None
        return env_cfg
    

def main():
    """Main function."""
    # Load trained JIT policy
    policy_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Checkpoint not found: {policy_path}")

    print(f"[INFO] Loading policy from: {policy_path}")
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy_model = torch.jit.load(file, map_location=args_cli.device)
    

    # Setup environment configuration
    env_cfg = DigitFlatEnvCfg_PLAY()
    # 3. ตั้งค่าพื้นฐานและตำแหน่งเกิด (Robot Origin)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 4.0
    
    # สำคัญ: ต้องเซตตรงนี้หลังจากสร้าง env_cfg แล้ว
    # x=0, y=0, z=1.2 (สูงจากพื้นป้องกันการทับ mesh)
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.2)
    env_cfg.curriculum = None

    # Configure command ranges to include zero velocity
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

    # Setup scene based on user selection
    env_cfg = setup_scene(env_cfg, args_cli.scene)

    # Device configuration
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    # Create environment
    print(f"[INFO] Creating environment with {env_cfg.scene.num_envs} robot(s)...")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Create interactive policy controller
    g1_policy = G1InteractivePolicy(policy_model, args_cli.device, use_gamepad=args_cli.use_gamepad)

    # Run simulation
    print("[INFO] Starting simulation. Use controls to move the robot.")
    obs, _ = env.reset()

    with torch.inference_mode():
        while simulation_app.is_running():
            # Compute action with current command
            action = g1_policy.compute_action(obs)
            # Step environment
            obs, _, _, _, _ = env.step(action)

    # Cleanup
    g1_policy.cleanup()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()