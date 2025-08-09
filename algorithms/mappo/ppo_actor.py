import numpy as np
import torch
import torch.nn as nn

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
from ..utils.utils import check


def safe_isnan(x):
    if isinstance(x, np.ndarray):
        return np.isnan(x).any()
    elif isinstance(x, torch.Tensor):
        return torch.isnan(x).any()
    else:
        return False


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOActor, self).__init__()
        # network config
        self.gain = args.gain
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)
        # (1) feature extraction module
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        # (2) rnn module
        input_size = self.base.output_size
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size
        # (3) act module
        self.act = ACTLayer(act_space, input_size, self.act_hidden_size, self.activation_id, self.gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, deterministic=False):
        if safe_isnan(obs):
            print('[NaN DEBUG] obs (forward):', obs)
        if safe_isnan(rnn_states):
            print('[NaN DEBUG] rnn_states (forward):', rnn_states)
        if safe_isnan(masks):
            print('[NaN DEBUG] masks (forward):', masks)
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)
        
        # NaN 检查：base 网络输出
        if safe_isnan(actor_features):
            print('[NaN DEBUG] actor_features (base output):', actor_features)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
            
            # NaN 检查：RNN 输出
            if safe_isnan(actor_features):
                print('[NaN DEBUG] actor_features (after RNN):', actor_features)

        actions, action_log_probs = self.act(actor_features, deterministic)

        if hasattr(self, 'logits') and safe_isnan(self.logits):
            print('[NaN DEBUG] logits (forward):', self.logits)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        if safe_isnan(obs):
            print('[NaN DEBUG] obs:', obs)
        if safe_isnan(rnn_states):
            print('[NaN DEBUG] rnn_states:', rnn_states)
        if safe_isnan(action):
            print('[NaN DEBUG] action:', action)
        if safe_isnan(masks):
            print('[NaN DEBUG] masks:', masks)
        if active_masks is not None and safe_isnan(active_masks):
            print('[NaN DEBUG] active_masks:', active_masks)
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)
        
        # NaN 检查：base 网络输出
        if safe_isnan(actor_features):
            print('[NaN DEBUG] actor_features (base output):', actor_features)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
            
            # NaN 检查：RNN 输出
            if safe_isnan(actor_features):
                print('[NaN DEBUG] actor_features (after RNN):', actor_features)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks)

        if hasattr(self, 'logits') and safe_isnan(self.logits):
            print('[NaN DEBUG] logits:', self.logits)
        return action_log_probs, dist_entropy
