import numpy as np
import mindspore
from model import Network

model = Network()

param_dict = mindspore.load_checkpoint("modelbest/model.ckpt")
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
print(param_not_load)

model.set_train(False)


def predict(data, mode):
    if mode == 'one':
        print(data)
        if not isinstance(data, mindspore.Tensor):
            data = mindspore.Tensor(data, dtype=mindspore
                                    .float32)
        pred = model(data)
        predicted = np.where(pred > .5, 1, 0)
        return predicted
    if mode == 'dataframe':
        pred_result = []
        if not isinstance(data, mindspore.Tensor):
            data = mindspore.Tensor(data, dtype=mindspore
                                    .float32)
            for row_data in data:
                pred = model(row_data)
                predicted = np.where(pred > .5, 1, 0)[0]
                pred_result.append(predicted)
            return pred_result


if __name__ == '__main__':
    result = predict([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], 'dataframe')
    print(result)
