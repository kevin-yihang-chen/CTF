import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class DAttention(nn.Module):
    """Single-head attention pooling over a bag of patch features."""

    def __init__(self, input_dim=512, dropout=0.25, act='relu'):
        super().__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        feature_layers = [nn.Linear(input_dim, 512)]
        feature_layers.append(nn.GELU() if act.lower() == 'gelu' else nn.ReLU())
        if dropout:
            feature_layers.append(nn.Dropout(0.25))
        self.feature = nn.Sequential(*feature_layers)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K),
        )

        self.apply(initialize_weights)

    def forward(self, x):
        feature = x.squeeze(0)
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)
        A = F.softmax(A, dim=-1)
        return torch.mm(A, feature)
