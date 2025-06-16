import numpy as np


class Diabetes_dataset:
    def __init__(self, root_data):
        self.root_data = root_data
        self._load_data()

    def _load_data(self):
        # 划分特征 标签
        feature_list = [x for x in self.root_data.columns if x != 'diabetes']
        label = 'diabetes'

        self.data = self.root_data[feature_list]
        self.label = self.root_data[label]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data.iloc[item].astype(np.float32), self.label.iloc[item].astype(np.float32)


def datapipe(dataset, batch_size):
    dataset = dataset.batch(batch_size)
    return dataset
