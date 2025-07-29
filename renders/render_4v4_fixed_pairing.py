import numpy as np
import torch
import sys
import os

# Manually add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.JSBSim.envs import IndepCombatEnv  # Use our custom Env
from algorithms.ppo.ppo_actor import PPOActor
from gymnasium import spaces
import logging
logging.basicConfig(level=logging.INFO)

# --- Device Setup ---
# Automatically detect and use MPS (Apple Silicon GPU) if available, otherwise fallback to CPU.
if torch.backends.mps.is_available():
    device = torch.device('mps')
    logging.info("Using MPS (Apple Silicon GPU) for acceleration.")
else:
    device = torch.device('cpu')
    logging.info("MPS not available, using CPU.")

class Args:
    """A dummy class to hold model configuration."""
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=device) # Using auto-detected device
        self.use_prior = False

def _t2n(x):
    """Convert a torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

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
        
        # Get the first and last time stamps
        first_time = time_stamps[0][1]  # e.g., "#0.20"
        last_time = time_stamps[-1][1]  # e.g., "#50.40"
        
        # Create static objects that appear at the first time stamp
        static_objects = []
        static_objects.append(first_time)  # Use the first time stamp instead of "#0.00"
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
            
        logging.info(f"Added {len(static_objects)-1} static objects to {filepath} at time {first_time}")
        
    except Exception as e:
        logging.error(f"Failed to add static objects: {e}")

# --- Configuration ---
NUM_AGENTS = 8
RENDER = True
# This is the path to your best 1v1 self-play model.
# All 8 agents will use this model.
MODEL_DIR = "/Users/zoumaoming/LAG/scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1_selfplay_from_baseline/run3"
MODEL_FILENAME = "actor_latest.pt"
OUTPUT_FILENAME = "4v4_fixed_pairing_render.txt.acmi"

# --- Static Objects Configuration ---
# Add static objects to the ACMI file for better visualization
ADD_STATIC_OBJECTS = True

# --- Environment and Policy Setup ---
logging.info("Setting up environment and policy...")
# Use the IndepCombatEnv with our custom 4v4 scenario
env = IndepCombatEnv("4v4/NoWeapon/IndepFixedPairing")
env.seed(1)  # Use a fixed seed for reproducibility
args = Args()

# For rendering and evaluation, we know the exact single-agent observation and action spaces
# that our pre-trained 1v1 model uses. We hardcode them here to ensure perfect compatibility.
single_agent_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
single_agent_act_space = spaces.MultiDiscrete([41, 41, 41, 30])

# Create a single policy instance using the correct single-agent spaces.
policy = PPOActor(args, single_agent_obs_space, single_agent_act_space, device=device)
policy.eval()

# Load the trained model weights
model_path = f"{MODEL_DIR}/{MODEL_FILENAME}"
logging.info(f"Loading model from: {model_path}")
policy.load_state_dict(torch.load(model_path, map_location=device))

# --- Main Render Loop ---
logging.info("Resetting environment...")
obs = env.reset()
if RENDER:
    env.render(mode='txt', filepath=OUTPUT_FILENAME)

# Initialize RNN states for all agents
rnn_states = np.zeros((NUM_AGENTS, 1, args.recurrent_hidden_size), dtype=np.float32)
masks = np.ones((NUM_AGENTS, 1), dtype=np.float32)
episode_rewards = 0

# --- 八靶点生成配置 ---
octpoint_records = []  # 存储每次插入八靶点的内容
octpoint_interval = 5.0  # 每隔5秒
octpoint_next_time = octpoint_interval
octpoint_id_base = 2000000  # 八靶点ID起始
dt = getattr(env, 'control_dt', 0.2)  # 仿真步长，默认0.2s

logging.info("Starting render loop...")
while True:
    all_actions = []
    # Loop through each agent to get its action individually
    for i in range(NUM_AGENTS):
        # Prepare single agent's data
        single_obs = np.expand_dims(obs[i], axis=0)
        single_rnn_state = np.expand_dims(rnn_states[i], axis=0)
        single_mask = np.expand_dims(masks[i], axis=0)

        # Convert to torch tensors and move to device
        torch_obs = torch.from_numpy(single_obs).float().to(device)
        torch_rnn_state = torch.from_numpy(single_rnn_state).float().to(device)
        torch_mask = torch.from_numpy(single_mask).float().to(device)

        # Get action from the policy for one agent
        action, _, new_rnn_state = policy(torch_obs, torch_rnn_state, torch_mask, deterministic=True)
        
        # Store the action and updated RNN state
        all_actions.append(_t2n(action))
        rnn_states[i] = _t2n(new_rnn_state)

    # Combine all actions into a single array for the environment step
    actions = np.concatenate(all_actions, axis=0)

    # Step the environment with the actions for all agents
    obs, rewards, dones, infos = env.step(actions)
    
    episode_rewards += rewards
    
    # --- 每隔5秒在红方飞机轨迹点生成八靶点 ---
    current_time = env.current_step * dt
    if current_time >= octpoint_next_time:
        # 红方前4个飞机 (假设ID为A0100, A0200, A0300, A0400)
        red_agent_ids = list(env.agents.keys())[:4]
        for idx, agent_id in enumerate(red_agent_ids):
            agent = env.agents[agent_id]
            # 获取飞机当前位置 (假设agent有lon, lat, alt属性)
            try:
                lon = agent.lon
                lat = agent.lat
                alt = agent.alt
            except AttributeError:
                # 如果agent没有这些属性，尝试从其他方式获取位置
                # 这里需要根据你的环境实际情况调整
                lon = 120.0 + idx * 0.01  # 默认位置
                lat = 60.0 + idx * 0.01
                alt = 5000.0
                
            oct_id = octpoint_id_base + idx + 1
            oct_line = f"{oct_id},T={lon:.6f}|{lat:.6f}|{alt:.2f},Type=Navaid+Static+Waypoint,Name=OctPoint{idx+1}"
            octpoint_records.append((current_time, oct_line))
            
        logging.info(f"在时间 {current_time:.1f}s 为红方飞机生成八靶点")
        octpoint_next_time += octpoint_interval
    
    if RENDER:
        env.render(mode='txt', filepath=OUTPUT_FILENAME)

    if dones.all():
        logging.info("All agents are done. Episode finished.")
        logging.info(f"Final Info: {infos}")
        break
        
    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    logging.info(f"Step: {env.current_step}, Health: {bloods}")

logging.info(f"Total Episode Rewards: {episode_rewards.sum()}")
logging.info(f"Render file saved to: {OUTPUT_FILENAME}")

# Add static objects to the ACMI file if enabled
if ADD_STATIC_OBJECTS:
    add_static_objects_to_acmi(OUTPUT_FILENAME)

# --- 插入八靶点到ACMI文件 ---
def insert_octpoints_to_acmi(filepath, octpoint_records):
    """将八靶点记录插入到ACMI文件中"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # 为每个八靶点找到合适的时间戳插入点
        for t, line in octpoint_records:
            t_str = f"#{t:.2f}"
            # 找到最接近的时间戳行
            for i, l in enumerate(lines):
                if l.strip().startswith('#'):
                    try:
                        if abs(float(l.strip()[1:]) - t) < 1e-2:
                            lines.insert(i+1, line+'\n')
                            break
                    except ValueError:
                        continue
        
        with open(filepath, 'w') as f:
            f.writelines(lines)
        logging.info(f"已插入 {len(octpoint_records)} 个八靶点到ACMI文件")
        
    except Exception as e:
        logging.error(f"插入八靶点时出错: {e}")

if octpoint_records:
    insert_octpoints_to_acmi(OUTPUT_FILENAME, octpoint_records) 