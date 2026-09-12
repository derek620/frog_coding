from datasets import load_from_disk
from torch.utils.data import DataLoader
from util.data import CriteoDataset
from const import BATCH_SIZE, FEATURE_NAMES

def load_data():
    raw_dataset = load_from_disk('./data/criteo_train_data_subset')
    sub_datasets = raw_dataset.train_test_split(test_size=0.2, seed=68)
    print(sub_datasets.shape)
    return sub_datasets['train'], sub_datasets['test']



train_dataset, test_dataset = load_data()


# t_dataset = train_dataset.with_format('torch')

# t_dataset = train_dataset.select_columns(FEATURE_NAMES).with_format('torch')

datas = CriteoDataset(train_dataset, FEATURE_NAMES)

    

train_loader = DataLoader(
    datas, 
    batch_size=BATCH_SIZE,
    shuffle=True
)

tdatas = CriteoDataset(test_dataset, FEATURE_NAMES)
test_loader = DataLoader(tdatas, batch_size=BATCH_SIZE, shuffle=True)


# import torch

# t = train_dataset.with_format('torch')
# for fname in FEATURE_NAMES:
#     x = torch.tensor(t[fname], dtype=torch.float32)
#     print("NAN= ", torch.isnan(x).any())
#     x = torch.nan_to_num(x, nan=0.0)
#     print("NAN= ", torch.isnan(x).any())
#     f1_mean = x.mean()
#     f1_min = x.min()
#     f1_max = x.max()
#     print(fname, ' ', x.dtype, x.shape)
#     print('min = ', f1_min)
#     print('max = ', f1_max)
#     print('mean = ', f1_mean)

