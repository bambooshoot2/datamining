from mindspore import nn


class ImprovedNetwork(nn.Cell):
    def __init__(self):
        super().__init__()
        self.dense_sequential = nn.SequentialCell(
            nn.Dense(8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Dense(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Dense(32, 1),
            nn.Sigmoid()
        )

    def construct(self, x):
        return self.dense_sequential(x).flatten()