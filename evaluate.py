
from sklearn.metrics import roc_auc_score
import torch
from features import test_loader
import numpy as np

def calc_auc(model):
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            # print("NAN= ", torch.isnan(x).any())
            # print("INF = ", torch.isfinite(x).any())
            x = x.float()
            logit = model(x)
            all_logits.append(logit.cpu())
            all_labels.append(y.cpu())
        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()
    
        # print("label= shape: ", all_labels.shape, np.isnan(all_labels).any())
        # print("logit= shape: ", all_logits.shape, np.isnan(all_logits).any())
        auc = roc_auc_score(all_labels, all_logits)

        return auc

