import copy
import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def get_shape_from_space(space):
    if isinstance(space, gym.spaces.Discrete):
        return (1,)
    elif isinstance(space, gym.spaces.Box) \
            or isinstance(space, gym.spaces.MultiDiscrete) \
            or isinstance(space, gym.spaces.MultiBinary):
        return space.shape
    elif isinstance(space,gym.spaces.Tuple) and \
           isinstance(space[0], gym.spaces.MultiDiscrete) and \
               isinstance(space[1], gym.spaces.Discrete):
        return (space[0].shape[0] + 1,)
    else:
        raise NotImplementedError(f"Unsupported action space type: {type(space)}!")


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def init(module: nn.Module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def safe_tensor_ops(tensor, operation, default_value=0.0):
    """
    Safely perform operations on tensors that might contain NaN values.
    
    Args:
        tensor: Input tensor
        operation: Function to apply to tensor
        default_value: Value to use if operation fails
    
    Returns:
        Result of operation or default_value if operation fails
    """
    try:
        if torch.isnan(tensor).any():
            return default_value
        return operation(tensor)
    except:
        return default_value

def clip_gradients_norm(model, max_norm=1.0):
    """
    Clip gradients by norm with additional NaN checking.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
    
    Returns:
        Total norm of parameters before clipping
    """
    total_norm = 0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            # Check for NaN gradients
            if torch.isnan(p.grad).any():
                p.grad.data.zero_()
                continue
                
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count == 0:
        return 0.0
        
    total_norm = total_norm ** (1. / 2)
    
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    
    return total_norm

def replace_nan_with_zero(tensor):
    """
    Replace NaN values in tensor with zeros.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Tensor with NaN values replaced by zeros
    """
    if isinstance(tensor, torch.Tensor):
        return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    elif isinstance(tensor, np.ndarray):
        return np.where(np.isnan(tensor), np.zeros_like(tensor), tensor)
    else:
        return tensor
