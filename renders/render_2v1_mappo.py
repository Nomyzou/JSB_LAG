#!/usr/bin/env python3
"""
2v1 MAPPO Training Render Script
Renders the trained 2v1 MAPPO model and generates ACMI file for visualization
"""

import os
import sys
import torch
import numpy as np
import logging
from gymnasium import spaces
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.mappo.ppo_actor import PPOActor
from envs.JSBSim.envs.mappo_2v1_env import MAPPOTraining2v1Env
# Local tensor-to-numpy helper
def _t2n(x):
    return x.detach().cpu().numpy()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Args:
    """Configuration class for the policy"""
    def __init__(self):
        self.gain = 0.01
        self.hidden_size = "128 128"
        self.act_hidden_size = "128 128"
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.use_prior = False
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))


def add_static_objects_to_acmi(filepath):
    """Add static objects (markers, lines, etc.) to the ACMI file for better visualization."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find all time stamps in the file
        time_stamps = []
        for i, line in enumerate(lines):
            if line.startswith('#') and not line.startswith('#0.00'):
                time_stamps.append((i, line.strip()))
        
        if not time_stamps:
            logging.warning("No time stamps found in ACMI file")
            return
        
        # Get the first time stamp
        first_time = time_stamps[0][1]  # e.g., "#0.20"
        
        # Create static objects that appear at the first time stamp
        static_objects = []
        static_objects.append(first_time)  # Use the first time stamp
        static_objects.append("StaticObject,T=120.0|60.0|0,Name=BattleFieldCenter,Color=Yellow,Shape=Sphere")
        static_objects.append("StaticObject,T=119.9|59.9|0,Name=SWCorner,Color=Orange,Shape=Cube")
        static_objects.append("StaticObject,T=120.1|59.9|0,Name=SECorner,Color=Orange,Shape=Cube")
        static_objects.append("StaticObject,T=119.9|60.1|0,Name=NWCorner,Color=Orange,Shape=Cube")
        static_objects.append("StaticObject,T=120.1|60.1|0,Name=NECorner,Color=Orange,Shape=Cube")
        static_objects.append("StaticObject,T=120.0|60.0|20000,Name=AltitudeRef,Color=Green,Shape=Sphere")
        static_objects.append("StaticObject,T=119.975|60.0|0,Name=TeamLine,Color=White,Shape=Line")
        
        # Insert static objects after the first time stamp
        insert_pos = time_stamps[0][0] + 1
        for i, obj in enumerate(static_objects):
            lines.insert(insert_pos + i, obj + '\n')
        
        # Write back to file
        with open(filepath, 'w') as f:
            f.writelines(lines)
        
        logging.info(f"Added static objects to {filepath}")
        
    except Exception as e:
        logging.error(f"Error adding static objects: {e}")


def main():
    parser = argparse.ArgumentParser(description="Render 2v1 MAPPO and generate ACMI")
    parser.add_argument("--model-dir", type=str, default="scripts/results/MAPPOTraining2v1/2v1/NoWeapon/MAPPOTraining/mappo/v1_2v1_mappo_training/wandb/latest-run/files", help="Directory containing saved actor/critic models")
    parser.add_argument("--episode", type=int, default=None, help="Episode number to load, e.g., 12499. If omitted, uses latest")
    parser.add_argument("--use-latest", action="store_true", help="Force using actor_latest.pt")
    parser.add_argument("--output", type=str, default="2v1_mappo_render.txt.acmi", help="Output ACMI filename")
    parser.add_argument("--scenario", type=str, default="2v1/NoWeapon/MAPPOTraining", help="Scenario name for the environment")
    parser.add_argument("--no-static", action="store_true", help="Do not add static objects to ACMI")
    parser.add_argument("--target-steps", type=int, default=400, help="Target number of steps to render (default: 400)")
    args_cli = parser.parse_args()
    
    RENDER = True
    MODEL_DIR = args_cli.model_dir
    OUTPUT_FILENAME = args_cli.output
    ADD_STATIC_OBJECTS = not args_cli.no_static
    SCENARIO = args_cli.scenario
    TARGET_STEPS = args_cli.target_steps
    
    # Determine model path with fallback
    def pick_latest_model(base_dir: str) -> str:
        preferred = os.path.join(base_dir, "actor2v1_latest.pt")
        fallback = os.path.join(base_dir, "actor_latest.pt")
        if os.path.exists(preferred):
            return preferred
        if os.path.exists(fallback):
            return fallback
        return None

    if args_cli.use_latest:
        candidate = pick_latest_model(MODEL_DIR)
        if candidate is not None:
            model_path = candidate
    elif args_cli.episode is not None:
        candidate = os.path.join(MODEL_DIR, f"actor_episode_{args_cli.episode}.pt")
        if os.path.exists(candidate):
            model_path = candidate
        else:
            logging.warning(f"actor_episode_{args_cli.episode}.pt not found, trying latest actor file")
            candidate_latest = pick_latest_model(MODEL_DIR)
            if candidate_latest is not None:
                model_path = candidate_latest
    else:
        candidate = pick_latest_model(MODEL_DIR)
        if candidate is not None:
            model_path = candidate
    
    if model_path is None:
        logging.error(f"No valid actor model file found under {MODEL_DIR}")
        return
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Environment setup
    logging.info("Setting up environment...")
    env = MAPPOTraining2v1Env(SCENARIO)
    env.seed(42)  # Fixed seed for reproducibility
    
    # Policy setup
    logging.info("Setting up policy...")
    args = Args()
    
    # Create policy with correct observation and action spaces
    obs_space = spaces.Box(low=-10., high=10., shape=(21,), dtype=np.float32)
    act_space = spaces.MultiDiscrete([41, 41, 41, 30])
    
    policy = PPOActor(args, obs_space, act_space, device=device)
    policy.eval()
    
    # Load the trained model
    logging.info(f"Loading model from: {model_path}")
    
    policy.load_state_dict(torch.load(model_path, map_location=device))
    logging.info("Model loaded successfully")
    
    # Main render loop
    logging.info("Starting render...")
    num_agents = env.num_agents
    obs, share_obs = env.reset()
    obs = obs.astype(np.float32)
    
    if RENDER:
        env.render(mode='txt', filepath=OUTPUT_FILENAME)
    
    # Initialize RNN states and masks
    rnn_states = np.zeros((num_agents, args.recurrent_hidden_layers, args.recurrent_hidden_size), dtype=np.float32)
    masks = np.ones((num_agents, 1), dtype=np.float32)
    episode_rewards = 0
    
    step_count = 0
    max_steps = TARGET_STEPS  # Target number of steps
    
    while step_count < max_steps:
        step_count += 1
        
        # Get actions from policy
        actions, _, rnn_states = policy(
            obs,
            rnn_states,
            masks,
            deterministic=True
        )

        actions = _t2n(actions)
        rnn_states = _t2n(rnn_states)
        
        # Take step in environment
        obs, share_obs, rewards, dones, infos = env.step(actions)
        obs = obs.astype(np.float32)
        # Update masks: 1 for alive, 0 for done
        masks = 1.0 - dones.astype(np.float32)
        
        # Accumulate rewards (only for red team agents)
        red_rewards = rewards[:, :2, ...]  # First 2 agents are red team
        episode_rewards += red_rewards
        
        # Render current state
        if RENDER:
            env.render(mode='txt', filepath=OUTPUT_FILENAME)
        
        # Check if episode is done
        if dones.all():
            logging.info(f"Episode finished at step {step_count}.")
            logging.info(f"Final Info: {infos}")
            # Continue rendering to reach target steps even if episode is done
            if step_count < max_steps:
                logging.info(f"Continuing to render until step {max_steps}...")
                # Keep rendering the final state for remaining steps
                for remaining_step in range(step_count + 1, max_steps + 1):
                    if RENDER:
                        env.render(mode='txt', filepath=OUTPUT_FILENAME)
                    step_count = remaining_step
                    if remaining_step % 50 == 0:
                        logging.info(f"Extended render step: {remaining_step}")
            break
        
        # Log progress
        if step_count % 50 == 0:
            bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
            logging.info(f"Step: {step_count}, Health: {bloods}")
    
    logging.info(f"Total steps rendered: {step_count}")
    logging.info(f"Total Episode Rewards: {episode_rewards.sum()}")
    logging.info(f"Render file saved to: {OUTPUT_FILENAME}")
    
    # Add static objects to the ACMI file if enabled
    if ADD_STATIC_OBJECTS:
        add_static_objects_to_acmi(OUTPUT_FILENAME)
    
    logging.info("Render completed successfully!")

if __name__ == "__main__":
    main() 