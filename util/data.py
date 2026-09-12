import torch
from torch.utils.data import Dataset

class CriteoDataset(Dataset):
    def __init__(self, hf_dataset, feature_names):
        self.dataset = hf_dataset
        self.features = feature_names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        x = torch.tensor(
            [row[name] if row[name] is not None else 0.0 for name in self.features], dtype=torch.float32
        )
        # x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0,neginf=0.0)

        y = torch.tensor(
            row['label'], dtype=torch.float32
        )

        return x, y
