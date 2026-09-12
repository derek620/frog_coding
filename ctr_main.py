import torch.nn as nn
import torch
from util.lr_model import CTRLogisticRegression
from const import FEATURE_NAMES
from features import train_loader
from evaluate import calc_auc


model = CTRLogisticRegression(len(FEATURE_NAMES))
opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
crit = nn.BCEWithLogitsLoss()


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


