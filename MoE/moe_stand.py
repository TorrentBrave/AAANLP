import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, top_k, expert_capacity, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList({Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)})
``        # self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.expert_capacity = expert_capacity
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        gate_weights = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.einsum('bse,bse->bs', gate_weights, expert_outputs)
        return output