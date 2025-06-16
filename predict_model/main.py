import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import dataset as ds
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from model import ImprovedNetwork
from load_data import Diabetes_dataset

# 设置随机种子保证可重复性
mindspore.set_seed(42)


def datapipe(dataset, batch_size):
    return dataset.batch(batch_size)


def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits


def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss


def train(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()
    total_loss = 0

    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)
        total_loss += loss.asnumpy()

        if batch % 100 == 0:
            current_loss = loss.asnumpy()
            print(f"批次: {batch:>3d}/{size:>3d} | 损失: {current_loss:>7f}")

    return total_loss / size


def test(model, dataset, loss_fn, best_f1):
    model.set_train(False)
    all_preds = []
    all_labels = []
    test_loss = 0
    num_batches = 0

    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        # 计算损失时使用原始输出(浮点数)
        batch_loss = loss_fn(pred, label.astype(mindspore.float32))
        test_loss += batch_loss.asnumpy()
        num_batches += 1

        # 转换为整数用于计算指标
        pred_labels = (pred > 0.5).astype(mindspore.int32)
        all_preds.append(pred_labels.asnumpy())
        all_labels.append(label.asnumpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    test_loss /= num_batches
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds)

    print(f"\n测试结果:")
    print(f"准确率: {(100 * accuracy):>0.1f}%")
    print(f"平均损失: {test_loss:>8f}")
    print(f"F1分数: {f1:>4f}")

    if f1 > best_f1:
        best_f1 = f1
        mindspore.save_checkpoint(model, "best_model.ckpt")
        print("\n保存了新的最佳模型(F1分数提升)")

    return best_f1


if __name__ == "__main__":
    # 数据加载和预处理
    pd.set_option('display.max_columns', None)
    data = pd.read_csv('diabetes_prediction_dataset.csv')

    # 特征工程和预处理
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    smoke_map = {'never': 0, 'former': 1, 'current': 2,
                 'not current': 3, 'ever': 4, 'No Info': 5}

    data['gender'] = data['gender'].map(gender_map)
    data['smoking_history'] = data['smoking_history'].map(smoke_map)

    # 处理缺失值
    data.fillna(data.median(), inplace=True)

    # 分割数据
    X = data.drop('diabetes', axis=1)
    y = data['diabetes']

    # 使用SMOTE处理类别不平衡
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # 特征归一化
    sclaer_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    scaler = MinMaxScaler()
    X_res[sclaer_columns] = scaler.fit_transform(X_res[sclaer_columns])

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42)

    # 创建数据集
    train_data = Diabetes_dataset(pd.concat([X_train, y_train], axis=1))
    test_data = Diabetes_dataset(pd.concat([X_test, y_test], axis=1))

    train_data = ds.GeneratorDataset(source=train_data, column_names=['feature', 'label'])
    test_data = ds.GeneratorDataset(source=test_data, column_names=['feature', 'label'])

    # 创建数据管道
    batch_size = 128
    train_dataset = datapipe(train_data, batch_size)
    test_dataset = datapipe(test_data, batch_size * 2)

    # 初始化模型
    model = ImprovedNetwork()

    # 损失函数(带类别权重)
    pos_weight = mindspore.Tensor([2.0])  # 根据类别分布调整
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 优化器
    initial_lr = 0.001
    optimizer = nn.Adam(model.trainable_params(), learning_rate=initial_lr)

    # 梯度函数
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # 训练循环
    epochs = 100
    best_f1 = 0
    patience = 5
    no_improve_epochs = 0

    print("开始训练...")
    for t in range(epochs):
        print(f"\n第 {t + 1}/{epochs} 轮训练")
        print("-------------------------------")

        avg_loss = train(model, train_dataset)
        print(f"\n第 {t + 1} 轮平均训练损失: {avg_loss:.4f}")

        current_f1 = test(model, test_dataset, loss_fn, best_f1)

        # 早停检查
        if current_f1 > best_f1:
            best_f1 = current_f1
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"\n第 {t + 1} 轮后触发早停")
                break

    print("\n训练完成!")
    print(f"达到的最佳F1分数: {best_f1:.4f}")