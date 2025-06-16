# 导入需要用到的库
# mindspore用于搭建简单的神经网络模型
import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import dataset as ds
# pandas用于读取数据
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from model import Network
from load_data import Diabetes_dataset


def datapipe(dataset, batch_size):
    dataset = dataset.batch(batch_size)
    return dataset


def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits


# 3. Define function of one-step training
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss


def train(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


def test(model, dataset, loss_fn, best_f1):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    sum_pred = []
    sum_label = []
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        pred = np.where(pred > 0.5, 1, 0)
        correct += (pred == label).sum()
        sum_pred.append(pred)
        sum_label.append(label)
    test_loss /= num_batches
    correct /= total
    f1_sum = f1_score([int(x) for x in sum_label[0]], sum_pred[0])
    print(
        f"Test: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}, F1 score:{f1_sum:>2f}  \n")
    if f1_sum > best_f1:
        best_f1 = f1_sum
        mindspore.save_checkpoint(model, "model.ckpt")
        print("----------------------------")
        print("Find new best f1_score model")
        print("Saved Model to model.ckpt")
        print("----------------------------")
    return best_f1


# 设置显示全部列
pd.set_option('display.max_columns', None)
# 读取数据
data = pd.read_csv('diabetes_prediction_dataset.csv')
# 简单的查看数据
# print(data)
# print(data.info())
# 对性别 吸烟史 两个类别变量进行转换
gender_map = {'Male': 0,
              'Female': 1,
              'Other': 2}
smoke_map = {'never': 0,
             'No Info': 1,
             'current': 2,
             'former': 3,
             'ever': 4,
             'not current': 5}
data['gender'] = data['gender'].map(gender_map)
data['smoking_history'] = data['smoking_history'].map(smoke_map)

#
sclaer_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
train_data, test_data = train_test_split(data, random_state=42)
scaler = MinMaxScaler()
train_data[sclaer_columns] = scaler.fit_transform(train_data[sclaer_columns])
test_data[sclaer_columns] = scaler.transform(test_data[sclaer_columns])
train_data = Diabetes_dataset(train_data)
test_data = Diabetes_dataset(test_data)
#
# # 查看转换是否成功
train_data = ds.GeneratorDataset(source=train_data, column_names=['feature', 'label'])
test_data = ds.GeneratorDataset(source=test_data, column_names=['feature', 'label'])

#
# train_data, test_data = data.split([0.8, 0.2])
#
train_dataset = datapipe(train_data, 128)
test_dataset = datapipe(test_data, 1024)

model = Network()
# print(model)

loss_fn = nn.BCELoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

# 2. Get gradient function
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

epochs = 5
f1_score_record = 0
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(model, train_dataset)
    f1_score_record = test(model, test_dataset, loss_fn, f1_score_record)
print("Done!")
