import numpy as np
from typing import Tuple, Dict, Any
from .env_base import BaseEnv
from ..tasks.mappo_2v1_training_task import MAPPOTraining2v1Task


class MAPPOTraining2v1Env(BaseEnv):
    """
    2v1 MAPPO Training Environment
    - Red team (2 aircraft): Use MAPPO for training
    - Blue team (1 aircraft): Use pre-trained 1v1 model
    """
    
    def __init__(self, config_name: str):
        super().__init__(config_name)
        self._create_records = False

    @property
    def share_observation_space(self):
        """Return shared observation space for MAPPO training"""
        return self.task.share_observation_space

    def load_task(self):
        """Load the 2v1 MAPPO training task"""
        taskname = getattr(self.config, 'task', None)
        if taskname == 'mappo_2v1_training':
            self.task = MAPPOTraining2v1Task(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
            share_obs (dict): {agent_id: initial state}
        """
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        return self._pack(obs), self._pack(share_obs)

    def reset_simulators(self):
        """Reset all simulators to initial conditions"""
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics.

        Args:
            action (np.ndarray): the agents' actions from MAPPO

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                share_obs: agents' share observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}

        # Apply actions for all agents (red team from MAPPO, blue team from pre-trained model)
        action = self._unpack(action)
        for agent_id in self.agents.keys():
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        
        # Run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        
        self.task.step(self)
        obs = self.get_obs()
        share_obs = self.get_state()

        # Calculate rewards
        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
        
        # For MAPPO training, we only care about red team rewards
        # Blue team rewards are not used for training
        ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
        for ego_id in self.ego_ids:
            rewards[ego_id] = [ego_reward]

        # Calculate termination conditions
        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        return self._pack(obs), self._pack(share_obs), self._pack(rewards), self._pack(dones), info

    def get_obs(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents"""
        obs = {}
        for agent_id in self.agents.keys():
            obs[agent_id] = self.task.get_obs(self, agent_id)
        return obs

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get shared state for MAPPO training.
        For MAPPO, all agents share the same global state.
        """
        # Get the global state from the task once.
        global_state = self.task.get_state(self)
        
        # All agents receive the same global state.
        state = {agent_id: global_state for agent_id in self.agents.keys()}
        return state 