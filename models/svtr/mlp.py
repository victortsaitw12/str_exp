import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0, init_cfg=None):
        super(MLP, self).__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H, W, C)`.
        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H, W, C)`.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x