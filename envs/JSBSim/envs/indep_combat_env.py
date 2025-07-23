import numpy as np
from .env_base import BaseEnv
from ..tasks import SingleCombatTask
from ..tasks.hierarchical_multiplecombat_task import HierarchicalMultipleCombatTask


class IndepCombatEnv(BaseEnv):
    """
    An independent learning environment for multi-agent combat, based on the SingleCombatEnv structure.
    It supports N-v-N scenarios but uses independent PPO learners for each agent.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'hierarchical_multiplecombat':
            self.task = HierarchicalMultipleCombatTask(self.config)
        else:
            # Fallback to single combat task if not specified
            self.task = SingleCombatTask(self.config)

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        # switch side
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        
        init_states = self.init_states.copy()
        # Randomly shuffle the initial states for all agents to ensure role randomization
        self.np_random.shuffle(init_states)
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(init_states[idx])
        self._tempsims.clear()
