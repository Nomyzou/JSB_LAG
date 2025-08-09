import torch
import torch.nn as nn

class ValueNorm(nn.Module):
    """
    A class for normalizing values, typically used for the value function in reinforcement learning.
    It maintains a running mean and standard deviation of the values it has seen.
    """
    def __init__(self, input_shape, device='cpu'):
        super(ValueNorm, self).__init__()
        self.device = device
        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(self.device)
        self.running_var = nn.Parameter(torch.ones(input_shape), requires_grad=False).to(self.device)
        self.count = nn.Parameter(torch.tensor(1e-4), requires_grad=False).to(self.device)

    def forward(self, x):
        # Calculate the running mean and variance
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count

        new_mean = self.running_mean + delta * batch_count / tot_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        # Update running stats
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.count.copy_(tot_count)

        # Normalize the input
        return (x - self.running_mean) / torch.sqrt(self.running_var + 1e-8)

    def state_dict(self):
        return {
            'running_mean': self.running_mean.cpu().numpy(),
            'running_var': self.running_var.cpu().numpy(),
            'count': self.count.cpu().numpy(),
        }

    def load_state_dict(self, state_dict):
        self.running_mean.copy_(torch.tensor(state_dict['running_mean'], device=self.device))
        self.running_var.copy_(torch.tensor(state_dict['running_var'], device=self.device))
        self.count.copy_(torch.tensor(state_dict['count'], device=self.device)) 