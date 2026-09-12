import torch.nn as nn
import torch

class CTRLogisticRegression(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1) # 自动创建w 和 b

    def forward(self, x):
        return self.linear(x).squeeze(-1) # 返回logit, 不做sigmod


from const import FEATURE_NAMES
model = CTRLogisticRegression(len(FEATURE_NAMES))
opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
crit = nn.BCEWithLogitsLoss()


from sklearn.metrics import roc_auc_score

def calc_auc(model):
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.float()
            logit = model(x)
            all_logits.append(logit.cpu())
            all_labels.append(y.cpu())
        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()

        auc = roc_auc_score(all_labels, all_logits)

        return auc





from features import train_loader
from features import test_loader

total_loss = 0.0
total_sample = 0
for epoch in range(200):
    model.train()
    for x, y in train_loader:
        logit = model(x)
        loss = crit(logit, y.float())
        opt.zero_grad()
        loss.backward()
        
        opt.step()
        total_loss += loss.item() * len(y)
        total_sample += len(y)

    avg_loss = total_loss / total_sample

    if epoch % 3 == 0:
        auc = calc_auc(model)
        print("idx = ", epoch, " loss = ", avg_loss, " auc = ", auc)


