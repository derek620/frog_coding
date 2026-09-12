import torch

# x:(B,D) Dense 特征
# y: (B, ) 0/1 label

D = 6
B = 1000

w = torch.zeros((1, D), requires_grad=True)
b = torch.zeros((B, 1), requires_grad=True)


def load_data():
    from datasets import load_from_disk
    raw_dataset = load_from_disk('./data/criteo_train_data_subset')
    sub_datasets = raw_dataset.train_test_split(test_size=0.2, seed=68)
    print(sub_datasets.shape)
    return sub_datasets['train'], sub_datasets['test']


def forward(x):
    z = x @ (w.T) + b
    return torch.sigmoid(z)


lr = 0.1


train_dataset, test_dataset = load_data()

print(len(train_dataset))

feature_names = ['label', 'integer_feature_1', 'integer_feature_2', 'integer_feature_3', 'integer_feature_4', 'integer_feature_5', 'integer_feature_6']

# t_dataset = train_dataset.select_columns(['label']).with_format('torch')





# for k, v in batch.items():
#     print(k, v.shape, v.dtype, v.device)


batch = next(iter(train_loader))
x = torch.stack(
    [batch['integer_feature_1'], 
    batch['integer_feature_2'],
    batch['integer_feature_3'],
    batch['integer_feature_4'],
    batch['integer_feature_5'],
    batch['integer_feature_6']
    ],
    dim=1
)
y = batch['label'].float().unsqueeze(1)


print("x has nan", torch.isnan(x).any())

x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0,neginf=0.0)
print("x has nan after norm", torch.isnan(x).any())

print(x.shape)
print(y.shape)


for step in range(200):

    p = forward(x)

    loss = -(y * (p + 1e-8).log() + (1-y) * (1-p + 1e-8).log()).mean()
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_(), b.grad.zero_()

    if step % 50 == 0:
        print(step, loss.item())

