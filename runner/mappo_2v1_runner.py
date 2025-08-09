import numpy as np
import torch
import time
import logging
import traceback
from typing import List
from runner.base_runner import Runner
from algorithms.utils.buffer import SharedReplayBuffer
from algorithms.utils.utils import check, get_gard_norm
# from algorithms.utils.utils import update_linear_schedule, huber_loss, mse_loss, soft_update, hard_update
from algorithms.utils.valuenorm import ValueNorm


def _t2n(x):
    return x.detach().cpu().numpy()


class MAPPOTraining2v1Runner(Runner):
    """
    Specialized runner for 2v1 MAPPO training, inspired by ShareJSBSimRunner.
    - Red team (2 aircraft): Use MAPPO for training
    - Blue team (1 aircraft): Use pre-trained 1v1 model (not trained)
    """
    
    def load(self):
        self.obs_space = self.envs.observation_space
        self.share_obs_space = self.envs.share_observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents  # 3 total agents
        self.num_red_agents = 2  # Red team agents (trained)
        self.num_blue_agents = 1  # Blue team agents (not trained)
        
        # episode step counter
        self.episode_steps = np.zeros(self.n_rollout_threads, dtype=np.int32)
        self.episode_lens_log = []
        
        # policy & algorithm
        if self.algorithm_name == "mappo":
            from algorithms.mappo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.mappo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
            
        self.policy = Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer - only for red team agents (2 agents)
        self.buffer = SharedReplayBuffer(self.all_args, self.num_red_agents, self.obs_space, self.share_obs_space, self.act_space)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        self.warmup()

        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads

        for episode in range(episodes):
            for step in range(self.buffer_size):
                # Sample actions for all agents
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)

                # Observe reward and next obs
                obs, share_obs, rewards, dones, infos = self.envs.step(actions)
                self.episode_steps += 1

                # Check for done environments and log episode lengths
                dones_squeezed = dones.squeeze(axis=-1)
                red_dones = np.all(dones_squeezed[:, :self.num_red_agents], axis=-1)
                blue_dones = np.all(dones_squeezed[:, self.num_red_agents:], axis=-1)
                dones_env = np.logical_or(red_dones, blue_dones)

                for i, done in enumerate(dones_env):
                    if done:
                        self.episode_lens_log.append(self.episode_steps[i])
                        self.episode_steps[i] = 0

                data = obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic

                # Insert data into buffer (only red team data)
                self.insert(data)

            # Compute return and update network
            self.compute()
            train_infos = self.train()

            # Post process
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads

            # Save model
            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)

            # Log information
            if episode % self.log_interval == 0:
                end = time.time()
                logging.info("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                             .format(self.all_args.scenario_name,
                                     self.algorithm_name,
                                     self.experiment_name,
                                     episode,
                                     episodes,
                                     self.total_num_steps,
                                     self.num_env_steps,
                                     int(self.total_num_steps / (end - start))))

                if len(self.episode_lens_log) > 0:
                    average_episode_length = np.mean(self.episode_lens_log)
                    logging.info(f"Average episode length: {average_episode_length:.2f} steps")
                    self.episode_lens_log = []
                else:
                    logging.info("No episodes finished during this interval.")

                if (self.buffer.masks == False).sum() > 0:
                    average_episode_rewards = self.buffer.rewards.sum() / (self.buffer.masks == False).sum()
                else:
                    average_episode_rewards = 0.0
                train_infos["average_episode_rewards"] = average_episode_rewards
                logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_info(train_infos, self.total_num_steps)

            # Eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(self.total_num_steps)

    def warmup(self):
        # Reset env
        obs, share_obs = self.envs.reset()
        
        # Extract red team data (first 2 agents) and populate buffer
        red_obs = obs[:, :self.num_red_agents, ...]
        red_share_obs = share_obs[:, :self.num_red_agents, ...]
        
        # Reshape the flattened share_obs to match the buffer's expected structure
        reshaped_share_obs = red_share_obs.reshape(self.n_rollout_threads, self.num_red_agents, self.num_agents, -1)
        self.buffer.share_obs[0] = reshaped_share_obs.copy()
        self.buffer.obs[0] = red_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        
        # The buffer only contains red team data, so we can use it directly
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                      np.concatenate(self.buffer.obs[step]),
                                      np.concatenate(self.buffer.rnn_states_actor[step]),
                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                      np.concatenate(self.buffer.masks[step]))
        
        # Split parallel data [N*M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # Pad actions to include blue team (the task will handle blue team actions)
        # The blue agent action is a placeholder, as it's controlled by a pre-trained model in the task.
        blue_actions = np.zeros((self.n_rollout_threads, self.num_blue_agents, *actions.shape[2:]), dtype=np.int32)
        all_actions = np.concatenate((actions, blue_actions), axis=1)

        return values, all_actions, action_log_probs, rnn_states_actor, rnn_states_critic

    @torch.no_grad()
    def compute(self):
        self.policy.prep_rollout()
        next_values = self.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                             np.concatenate(self.buffer.rnn_states_critic[-1]),
                                             np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.buffer.n_rollout_threads))
        self.buffer.compute_returns(next_values)

    def insert(self, data: List[np.ndarray]):
        obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data
        dones = dones.squeeze(axis=-1)

        # 2v1 episode终止逻辑
        red_dones = np.all(dones[:, :self.num_red_agents], axis=-1)
        blue_dones = np.all(dones[:, self.num_red_agents:], axis=-1)
        dones_env = np.logical_or(red_dones, blue_dones)

        # Reset RNN states for done environments
        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)

        # Create masks for buffer
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # 只提取红队数据
        red_obs = obs[:, :self.num_red_agents, ...]
        red_share_obs = share_obs[:, :self.num_red_agents, ...]
        red_actions = actions[:, :self.num_red_agents, ...]
        red_rewards = rewards[:, :self.num_red_agents, ...]
        red_masks = masks[:, :self.num_red_agents, ...]
        red_active_masks = active_masks[:, :self.num_red_agents, ...]
        red_rnn_states_actor = rnn_states_actor[:, :self.num_red_agents, ...]
        red_rnn_states_critic = rnn_states_critic[:, :self.num_red_agents, ...]

        # Reshape share_obs
        reshaped_share_obs = red_share_obs.reshape(self.n_rollout_threads, self.num_red_agents, self.num_agents, -1)

        # 调用 SharedReplayBuffer.insert，参数顺序严格对应
        self.buffer.insert(
            red_obs,                # obs
            reshaped_share_obs,     # share_obs
            red_actions,            # actions
            red_rewards,            # rewards
            red_masks,              # masks
            action_log_probs,       # action_log_probs
            values,                 # value_preds
            red_rnn_states_actor,   # rnn_states_actor
            red_rnn_states_critic,  # rnn_states_critic
            active_masks=red_active_masks
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        # eval method is not the priority now, keeping it as is.
        pass

    @torch.no_grad()
    def render(self):
        # render method is not the priority now, keeping it as is.
        pass

    def save(self, episode):
        """Save the model"""
        self.policy.save(self.save_dir, episode)

    def restore(self):
        """Restore the model"""
        self.policy.restore(self.model_dir, actor_filename='actor2v1_latest.pt', critic_filename='critic2v1_latest.pt') 