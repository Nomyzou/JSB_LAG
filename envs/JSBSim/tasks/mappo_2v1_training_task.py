import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple

from algorithms.ppo.ppo_actor import PPOActor
from envs.JSBSim.tasks.task_base import BaseTask
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import parse_config, LLA2NEU, NEU2LLA, get_AO_TA_R
from envs.JSBSim.termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn


def get_root_dir():
    """
    Finds the project root directory by searching upwards for a directory containing the 'LICENSE' file.
    """
    current_path = os.path.abspath(os.path.dirname(__file__))
    while not os.path.exists(os.path.join(current_path, 'LICENSE')):
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
        if parent_path == current_path:
            raise Exception("Could not find project root directory. Looking for 'LICENSE' file.")
        current_path = parent_path
    return current_path

class _Dict2Class(object):
    """A helper class to convert a dictionary to a class object recursively."""
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, _Dict2Class(value))
            else:
                setattr(self, key, value)

    def items(self):
        return self.__dict__.items()

    def get(self, key, default=None):
        return getattr(self, key, default)


class MAPPOTraining2v1Task(BaseTask):
    """
    2v1 MAPPO Training Task
    - Red team (2 aircraft): Use MAPPO for training
    - Blue team (1 aircraft): Use pre-trained 1v1 model
    """
    
    def __init__(self, config):
        config_dict = {key: getattr(config, key) for key in dir(config) if not key.startswith('__')}
        config_obj = _Dict2Class(config_dict)
        
        self.agent_num = config_obj.env.agent_num
        
        super().__init__(config_obj)
        
        # 设置终止条件
        self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]
        
        self.side_name = ['red', 'blue']
        self.aircraft_type = ['f16', 'f16']
        self._load_pretrained_model()
        self._blue_rnn_states = {}

    def _load_pretrained_model(self):
        self.pretrain_actor = None
        model_relative_path = self.config.aircraft_configs.B0100.get('pretrained_model_path')
        if not model_relative_path:
            return

        root_dir = get_root_dir()
        model_path = os.path.join(root_dir, model_relative_path)
        model_path_in_scripts = os.path.join(root_dir, 'scripts', model_relative_path)
        
        final_model_path = None
        if os.path.exists(model_path):
            final_model_path = model_path
        elif os.path.exists(model_path_in_scripts):
            final_model_path = model_path_in_scripts
        else:
            raise FileNotFoundError(f"Pre-trained model not found. Checked: {model_path} and {model_path_in_scripts}")

        class Args:
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
        args = Args()
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,))
        act_space = spaces.MultiDiscrete([41, 41, 41, 30])
        self.pretrain_actor = PPOActor(args, obs_space, act_space)
        
        state_dict = torch.load(final_model_path, map_location=self.config.simulation.device, weights_only=True)
        self.pretrain_actor.load_state_dict(state_dict)
        self.pretrain_actor.eval()
        print(f"Successfully constructed actor and loaded weights from {final_model_path}")

    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg, c.position_lat_geod_deg, c.position_h_sl_m,
            c.attitude_roll_rad, c.attitude_pitch_rad, c.attitude_heading_true_rad,
            c.velocities_v_north_mps, c.velocities_v_east_mps, c.velocities_v_down_mps,
            c.velocities_u_mps, c.velocities_v_mps, c.velocities_w_mps, c.velocities_vc_mps,
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm, c.fcs_elevator_cmd_norm,
            c.fcs_rudder_cmd_norm, c.fcs_throttle_cmd_norm,
        ]

    def load_observation_space(self):
        self.obs_length = 21
        self.observation_space = spaces.Box(low=-10., high=10., shape=(self.obs_length,), dtype=np.float32)
        self.share_observation_space = spaces.Box(low=-10., high=10., shape=(self.agent_num, self.obs_length), dtype=np.float32)

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    @property
    def num_agents(self) -> int:
        return self.config.env.agent_num

    def get_obs(self, env, agent_id):
        """Get observation for a specific agent."""
        agent = env.agents[agent_id]
        other_agents = [a for a in env.agents.values() if a.uid != agent_id]
        
        # Get ego agent state
        ego_state = np.array(agent.get_property_values(self.state_var))
        
        # NaN 检查和处理：ego 状态
        if np.isnan(ego_state).any():
            # 静默处理NaN，使用更合理的默认值
            default_ego_state = np.array([120.0, 60.0, 5000.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            ego_state = np.where(np.isnan(ego_state), default_ego_state, ego_state)
        
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        
        # NaN 检查和处理：ego 特征
        if np.isnan(ego_feature).any():
            # 静默处理NaN，使用更合理的默认值
            default_ego_feature = np.array([0.0, 0.0, 0.0, 200.0, 0.0, 0.0])
            ego_feature = np.where(np.isnan(ego_feature), default_ego_feature, ego_feature)
        
        # Initialize observation array
        norm_obs = np.zeros(self.observation_space.shape[0])
        
        # Ego agent features (first 9 elements)
        norm_obs[0] = ego_feature[0] / 10000  # N position
        norm_obs[1] = ego_feature[1] / 10000  # E position
        norm_obs[2] = ego_feature[2] / 1000   # U position
        norm_obs[3] = ego_feature[3] / 340    # Velocity
        norm_obs[4] = ego_feature[4] / 180    # Pitch
        norm_obs[5] = ego_feature[5] / 180    # Roll
        norm_obs[6] = ego_state[9] / 180      # Heading
        norm_obs[7] = ego_state[10] / 180     # Angle of attack
        norm_obs[8] = ego_state[11] / 180     # Sideslip angle

        # Other agents features (remaining elements)
        offset = 8
        for other_agent in other_agents:
            state = np.array(other_agent.get_property_values(self.state_var))
            
            # NaN 检查和处理：其他智能体状态
            if np.isnan(state).any():
                # 静默处理NaN，使用更合理的默认值
                default_state = np.array([120.0, 60.0, 5000.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                state = np.where(np.isnan(state), default_state, state)
            
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            
            # NaN 检查和处理：其他智能体特征
            if np.isnan(feature).any():
                # 静默处理NaN，使用更合理的默认值
                default_feature = np.array([0.0, 0.0, 0.0, 200.0, 0.0, 0.0])
                feature = np.where(np.isnan(feature), default_feature, feature)
            
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            
            # NaN 检查和处理：AO_TA_R 计算结果
            if np.isnan([AO, TA, R, side_flag]).any():
                # 静默处理NaN，使用合理的默认值
                AO = 0.0 if np.isnan(AO) else AO
                TA = 0.0 if np.isnan(TA) else TA
                R = 10000.0 if np.isnan(R) else R
                side_flag = 0.0 if np.isnan(side_flag) else side_flag
            
            offset += 1
            norm_obs[offset] = (state[9] - ego_state[9]) / 340
            offset += 1
            norm_obs[offset] = (state[2] - ego_state[2]) / 1000
            offset += 1
            norm_obs[offset] = AO
            offset += 1
            norm_obs[offset] = TA
            offset += 1
            norm_obs[offset] = R / 10000
            offset += 1
            norm_obs[offset] = side_flag

        # NaN 检查和处理：最终观测
        if np.isnan(norm_obs).any():
            # 静默处理NaN，使用合理的默认值
            norm_obs = np.where(np.isnan(norm_obs), 0.0, norm_obs)
        
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        
        # 最终统计信息
        if np.isnan(norm_obs).any():
            # 最后一次替换 NaN 值
            norm_obs = np.where(np.isnan(norm_obs), 0.0, norm_obs)
        else:
            # 只在调试模式下打印统计信息
            pass
        
        return norm_obs

    def _get_single_obs(self, env, agent_id):
        ego_agent = env.agents[agent_id]
        if not ego_agent.is_alive:
            return np.zeros(self.obs_length)
        
        norm_obs = np.zeros(self.obs_length)
        ego_state = np.array(ego_agent.get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        
        norm_obs[0] = ego_state[2] / 5000
        norm_obs[1] = np.sin(ego_state[3])
        norm_obs[2] = np.cos(ego_state[3])
        norm_obs[3] = np.sin(ego_state[4])
        norm_obs[4] = np.cos(ego_state[4])
        norm_obs[5] = ego_state[9] / 340
        norm_obs[6] = ego_state[10] / 340
        norm_obs[7] = ego_state[11] / 340
        norm_obs[8] = ego_state[12] / 340

        offset = 8
        other_agents = [sim for sim_id, sim in env.agents.items() if sim_id != agent_id and sim.is_alive]
        for other_agent in other_agents:
            state = np.array(other_agent.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            offset += 1
            norm_obs[offset] = (state[9] - ego_state[9]) / 340
            offset += 1
            norm_obs[offset] = (state[2] - ego_state[2]) / 1000
            offset += 1
            norm_obs[offset] = AO
            offset += 1
            norm_obs[offset] = TA
            offset += 1
            norm_obs[offset] = R / 10000
            offset += 1
            norm_obs[offset] = side_flag
        
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return np.nan_to_num(norm_obs, nan=0.0, posinf=10.0, neginf=-10.0)
    
    def get_reward(self, env, agent_id: str, info: dict) -> Tuple[float, dict]:
        """
        A more sophisticated reward function for 2v1 air combat.
        """
        ego_agent = env.agents[agent_id]
        if not ego_agent.is_alive:
            # Penalize for being shot down or crashing
            return -100.0, info

        # Constants for reward shaping
        R_MAX = 50000.0  # Max distance for reward calculation
        ALT_REWARD_SCALAR = 0.1
        DISTANCE_REWARD_SCALAR = 1.0
        ANGLE_REWARD_SCALAR = 2.0

        # Base survival reward
        reward = 0.1

        # Find the single blue agent
        blue_agent = None
        for agent in env.agents.values():
            if agent.uid.startswith('B') and agent.is_alive:
                blue_agent = agent
                break
        
        # If blue agent is down, red team wins
        if blue_agent is None:
            return 100.0, info

        # Calculate rewards based on state relative to the blue agent
        ego_state = np.array(ego_agent.get_property_values(self.state_var))
        blue_state = np.array(blue_agent.get_property_values(self.state_var))

        # 1. Altitude difference reward
        ego_alt = ego_state[2]
        blue_alt = blue_state[2]
        alt_diff = np.abs(ego_alt - blue_alt)
        # Reward for being at a similar altitude, scaled to be significant
        reward += (1 - alt_diff / R_MAX) * ALT_REWARD_SCALAR

        # 2. Distance reward
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        blue_cur_ned = LLA2NEU(*blue_state[:3], env.center_lon, env.center_lat, env.center_alt)
        distance = np.linalg.norm(np.array(ego_cur_ned) - np.array(blue_cur_ned))
        # Reward for getting closer to the enemy
        if distance < R_MAX:
            reward += (1 - distance / R_MAX) * DISTANCE_REWARD_SCALAR

        # 3. Angle reward (Aspect Angle and Tail Angle)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        blue_feature = np.array([*blue_cur_ned, *(blue_state[6:9])])
        AO, TA, R, _ = get_AO_TA_R(ego_feature, blue_feature, return_side=True)
        
        # Reward for being in an advantageous position (small AO, small TA)
        # We use cosine to make the reward highest at 0 degrees
        angle_reward = (np.cos(AO) + np.cos(TA)) / 2.0
        reward += angle_reward * ANGLE_REWARD_SCALAR

        return reward, info

    def get_state(self, env):
        all_obs = [self._get_single_obs(env, agent_id) for agent_id in sorted(env.agents.keys())]
        state = np.stack(all_obs, axis=0)
        return state

    def normalize_action(self, env, agent_id, action):
        if agent_id.startswith('A'):
            return self._normalize_mappo_action(action)
        else:
            if self.pretrain_actor is None:
                return np.array([0.0, 0.0, 0.0, 0.7])
            return self._normalize_blue_action(env, agent_id)

    def _normalize_mappo_action(self, action):
        action_continuous = np.array(action)
        norm_act = np.zeros(4)
        norm_act[0] = action_continuous[0] / 20. - 1.
        norm_act[1] = action_continuous[1] / 20. - 1.
        norm_act[2] = action_continuous[2] / 20. - 1.
        norm_act[3] = action_continuous[3] / 58. + 0.4
        return norm_act

    def _normalize_blue_action(self, env, agent_id):
        blue_agent = env.agents[agent_id]
        
        closest_red = None
        min_distance = float('inf')
        
        for red_id in ['A0100', 'A0200']:
            if red_id in env.agents and env.agents[red_id].is_alive:
                red_agent = env.agents[red_id]
                distance = np.linalg.norm(
                    np.array(blue_agent.get_property_values([c.position_long_gc_deg, c.position_lat_geod_deg, c.position_h_sl_ft])) -
                    np.array(red_agent.get_property_values([c.position_long_gc_deg, c.position_lat_geod_deg, c.position_h_sl_ft]))
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_red = red_agent
        
        if closest_red is None:
            return np.array([0.0, 0.0, 0.0, 0.7])
        
        low_level_obs = self._get_low_level_obs(blue_agent, closest_red, env)
        
        if agent_id not in self._blue_rnn_states:
            self._blue_rnn_states[agent_id] = torch.zeros((1, self.pretrain_actor.recurrent_hidden_size))
            
        masks = torch.ones((1, 1))
        
        actions, _, self._blue_rnn_states[agent_id] = self.pretrain_actor(
            low_level_obs, self._blue_rnn_states[agent_id], masks, deterministic=True
        )
        
        action_continuous = actions.detach().cpu().numpy().squeeze(0)
        
        norm_act = np.zeros(4)
        norm_act[0] = action_continuous[0] / 20. - 1.
        norm_act[1] = action_continuous[1] / 20. - 1.
        norm_act[2] = action_continuous[2] / 20. - 1.
        norm_act[3] = action_continuous[3] / 58. + 0.4
        return norm_act

    def _get_low_level_obs(self, ego_agent, enm_agent, env):
        norm_obs = np.zeros(15)
        state_var_for_blue = [
            c.position_long_gc_deg, c.position_lat_geod_deg, c.position_h_sl_m,
            c.attitude_roll_rad, c.attitude_pitch_rad, c.attitude_heading_true_rad,
            c.velocities_v_north_mps, c.velocities_v_east_mps, c.velocities_v_down_mps,
            c.velocities_u_mps, c.velocities_v_mps, c.velocities_w_mps, c.velocities_vc_mps
        ]
        ego_obs_list = np.array(ego_agent.get_property_values(state_var_for_blue))
        enm_obs_list = np.array(enm_agent.get_property_values(state_var_for_blue))
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        
        ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        
        return torch.from_numpy(np.expand_dims(np.clip(norm_obs, -10, 10), axis=0)).float()

    def reset(self, env):
        if self.pretrain_actor:
            self._blue_rnn_states = {agent_id: torch.zeros((1, self.pretrain_actor.recurrent_hidden_size)) for agent_id in env.agents.keys() if agent_id.startswith('B')}
        else:
            self._blue_rnn_states = {}
        return super().reset(env) 