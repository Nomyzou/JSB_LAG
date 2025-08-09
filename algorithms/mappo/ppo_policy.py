import numpy as np
import torch
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic


def safe_isnan(x):
    if isinstance(x, np.ndarray):
        return np.isnan(x).any()
    elif isinstance(x, torch.Tensor):
        return torch.isnan(x).any()
    else:
        return False


class PPOPolicy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):

        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr
        
        self.obs_space = obs_space
        self.cent_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = PPOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = PPOCritic(args, self.cent_obs_space, self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=self.lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Returns:
            values
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, active_masks=None):
        if safe_isnan(cent_obs):
            print('[NaN DEBUG] cent_obs:', cent_obs)
        if safe_isnan(obs):
            print('[NaN DEBUG] obs:', obs)
        if safe_isnan(rnn_states_actor):
            print('[NaN DEBUG] rnn_states_actor:', rnn_states_actor)
        if safe_isnan(rnn_states_critic):
            print('[NaN DEBUG] rnn_states_critic:', rnn_states_critic)
        if safe_isnan(action):
            print('[NaN DEBUG] action:', action)
        if safe_isnan(masks):
            print('[NaN DEBUG] masks:', masks)
        if active_masks is not None and safe_isnan(active_masks):
            print('[NaN DEBUG] active_masks:', active_masks)
        if hasattr(self, 'logits') and safe_isnan(self.logits):
            print('[NaN DEBUG] logits:', self.logits)
        # 正确补充 values、action_log_probs、dist_entropy 的定义
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """
        Returns:
            actions, rnn_states_actor
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    def save(self, save_dir, episode):
        """Save actor and critic models."""
        torch.save(self.actor.state_dict(), str(save_dir) + "/actor_episode_" + str(episode) + ".pt")
        torch.save(self.critic.state_dict(), str(save_dir) + "/critic_episode_" + str(episode) + ".pt")

    def restore(self, model_dir, actor_filename='actor_latest.pt', critic_filename='critic_latest.pt'):
        """Restore actor and critic models."""
        actor_state_dict = torch.load(str(model_dir) + '/' + actor_filename)
        self.actor.load_state_dict(actor_state_dict)
        critic_state_dict = torch.load(str(model_dir) + '/' + critic_filename)
        self.critic.load_state_dict(critic_state_dict)

    def copy(self):
        return PPOPolicy(self.args, self.obs_space, self.act_space, self.device)
