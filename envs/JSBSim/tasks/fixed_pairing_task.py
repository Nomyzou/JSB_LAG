import numpy as np
from gymnasium import spaces
from typing import Tuple
import torch

from ..tasks import SingleCombatTask
from ..core.catalog import Catalog as c
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor


class FixedPairingTask(SingleCombatTask):
    """
    A task for N-v-N combat where each agent is permanently paired with a specific enemy.
    It uses a pre-trained 1v1 model for maneuver execution and does not involve high-level learning.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Load the pre-trained 1v1 low-level policy
        self.lowlevel_policy = BaselineActor()
        # IMPORTANT: Make sure this path points to your trained 1v1 model
        model_path = get_root_dir() + '/results/SingleCombat/1v1/NoWeapon/Selfplay/v1_selfplay_from_baseline/run1/models/actor_latest.pt'
        self.lowlevel_policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.lowlevel_policy.eval()

    def load_action_space(self):
        # No actions are needed from the RL algorithm, but we define a dummy one to fit the API.
        self.action_space = spaces.Discrete(1)

    def normalize_action(self, env, agent_id, action):
        """
        Ignores the input action and uses a fixed pairing rule to select the target.
        Then, uses the 1v1 model to get the maneuver.
        """
        ego_agent = env.agents[agent_id]
        
        # --- FIXED PAIRING LOGIC ---
        # Extracts the numeric part of the agent's ID (e.g., '0100' from 'A0100')
        agent_index_str = agent_id[1:]
        # Creates the corresponding enemy ID (e.g., 'B0100')
        target_id = 'B' + agent_index_str
        
        # Find the target in the enemies list. This is more robust than assuming a fixed order.
        target_enemy = None
        for enemy in ego_agent.enemies:
            if enemy.uid == target_id:
                target_enemy = enemy
                break
        
        if target_enemy is None:
            # Fallback: if the paired enemy is dead or not found, target the first available enemy.
            target_enemy = next((en for en in ego_agent.enemies if en.is_alive), ego_agent.enemies[0])

        # Generate observation for the low-level policy (1v1 model)
        low_level_obs = self._get_low_level_obs(ego_agent, target_enemy, env)
        
        # Get action from the pre-trained 1v1 policy
        _action, self._inner_rnn_states[agent_id] = self.lowlevel_policy(low_level_obs, self._inner_rnn_states[agent_id])
        action_continuous = _action.detach().cpu().numpy().squeeze(0)
        
        # Normalize the continuous action for the simulator
        norm_act = np.zeros(4)
        norm_act[0] = action_continuous[0] / 20. - 1.
        norm_act[1] = action_continuous[1] / 20. - 1.
        norm_act[2] = action_continuous[2] / 20. - 1.
        norm_act[3] = action_continuous[3] / 58. + 0.4
        return norm_act

    def _get_low_level_obs(self, ego_agent, enm_agent, env):
        """
        Helper function to create observation for the 1v1 low-level policy.
        This must match the observation format that the 1v1 model was trained on.
        """
        norm_obs = np.zeros(15)
        ego_obs_list = np.array(ego_agent.get_property_values(self.state_var))
        enm_obs_list = np.array(enm_agent.get_property_values(self.state_var))
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
        
        return np.expand_dims(np.clip(norm_obs, -10, 10), axis=0)


    def reset(self, env):
        """Task-specific reset, including RNN states for the low-level policy.
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)

