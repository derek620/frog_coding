from datasets import load_dataset
from datasets import Dataset

train_ds = load_dataset("criteo/CriteoClickLogs", split='train', streaming=True)

mini_t_ds = train_ds.take(100 * 1000)

final_ds = Dataset.from_generator(lambda: mini_t_ds)

final_ds.save_to_disk('./data/criteo_train_data_subset')

# print(final_ds.shape())

