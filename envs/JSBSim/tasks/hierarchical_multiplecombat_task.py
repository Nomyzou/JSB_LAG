import numpy as np
from gymnasium import spaces
from typing import Tuple
import torch

from ..tasks import SingleCombatTask
from ..core.catalog import Catalog as c
from ..core.simulatior import MissileSimulator
from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward, MissilePostureReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor


class HierarchicalMultipleCombatTask(SingleCombatTask):
    """
    A hierarchical task for 4v4 combat.
    The high-level policy chooses a target, and the low-level policy (a pre-trained 1v1 model) executes the combat maneuver.
    """
    def __init__(self, config):
        super().__init__(config)
        self.team_size = 4

        # Load the pre-trained 1v1 low-level policy
        self.lowlevel_policy = BaselineActor()
        # IMPORTANT: Make sure this path points to your trained 1v1 model
        model_path = get_root_dir() + '/results/SingleCombat/1v1/NoWeapon/vsBaseline/v1/run1/models/actor_latest.pt'
        self.lowlevel_policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.lowlevel_policy.eval()

    @property
    def num_agents(self) -> int:
        return self.team_size * 2

    def load_observation_space(self):
        # Observation for the high-level policy (commander)
        # ego info (9) + other_agents_info (7 * 6) = 51
        self.obs_length = 9 + (self.num_agents - 1) * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        # High-level action: choose one of the 4 enemies to target
        self.action_space = spaces.Discrete(self.team_size)

    def get_obs(self, env, agent_id):
        """
        Provides observation for the high-level policy.
        This includes ego state and relative information of all other aircraft.
        """
        ego_agent = env.agents[agent_id]
        norm_obs = np.zeros(self.obs_length)
        
        # (1) Ego info normalization
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

        # (2) Relative info w.r.t partners and enemies
        offset = 9
        all_other_agents = ego_agent.partners + ego_agent.enemies
        for other_agent in all_other_agents:
            if other_agent.is_alive:
                state = np.array(other_agent.get_property_values(self.state_var))
                cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
                feature = np.array([*cur_ned, *(state[6:9])])
                AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
                norm_obs[offset] = (state[9] - ego_state[9]) / 340
                norm_obs[offset+1] = (state[2] - ego_state[2]) / 1000
                norm_obs[offset+2] = AO
                norm_obs[offset+3] = TA
                norm_obs[offset+4] = R / 10000
                norm_obs[offset+5] = side_flag
            offset += 6
        
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """
        Convert high-level action (target index) into low-level maneuver.
        """
        ego_agent = env.agents[agent_id]
        # The high-level action is the index of the chosen enemy
        target_enemy = ego_agent.enemies[action]

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

