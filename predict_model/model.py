from mindspore import nn


class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(8, 16),
            nn.ReLU(),
            nn.Dense(16, 16),
            nn.ReLU(),
            nn.Dense(16, 1),
            nn.Sigmoid()
        )

    def construct(self, x):
        x = self.dense_relu_sequential(x)
        return x.flatten()
