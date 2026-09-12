import torch.nn as nn

class CTRLogisticRegression(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1) # 自动创建w 和 b

    def forward(self, x):
        return self.linear(x).squeeze(-1) # 返回logit, 不做sigmod