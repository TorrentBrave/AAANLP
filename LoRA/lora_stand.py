import torch.nn as nn
import torch.nn.functional as F
import math

class LoRA(nn.Module):
    def __init__(self, input_dim, output_dim, rank=4):
        super().__init__()
        self.rank = rank
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize the low-rank matrices
        self.A = nn.Parameter(torch.randn(input_dim, rank))
        self.B = nn.Parameter(torch.randn(rank, output_dim))

        # Initialize the scaling factor
        self.scaling = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Compute the low-rank approximation
        low_rank_output = torch.matmul(x, self.A)
        low_rank_output = torch.matmul(low_rank_output, self.B)

        # Scale the output
        scaled_output = low_rank_output * self.scaling

        return scaled_output