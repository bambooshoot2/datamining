import argparse
import os
import pandas as pd
import numpy as np
import joblib
import mindspore
from mindspore import load_checkpoint, load_param_into_net
from model import ImprovedNetwork
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def print_data_format_info():
    """打印数据格式要求信息"""
    print("\n" + "=" * 80)
    print("测试数据格式要求:")
    print("文件应为CSV格式，包含以下列：")
    print("  - gender: 性别 (Female/Male/Other)")
    print("  - age: 年龄 (浮点数)")
    print("  - hypertension: 是否有高血压 (0/1)")
    print("  - heart_disease: 是否有心脏病 (0/1)")
    print("  - smoking_history: 吸烟史 (never/former/current/not current/ever/No Info)")
    print("  - bmi: 体重指数 (浮点数)")
    print("  - HbA1c_level: 糖化血红蛋白水平 (浮点数)")
    print("  - blood_glucose_level: 血糖水平 (浮点数)")
    print("  - diabetes: 是否患有糖尿病 (0/1，这是目标标签)")
    print("=" * 80 + "\n")


def preprocess_data(data_file, scaler_path):
    """预处理测试数据"""
    try:
        data = pd.read_csv(data_file)

        # 验证必要的列是否存在
        required_columns = ['gender', 'age', 'hypertension', 'heart_disease',
                            'smoking_history', 'bmi', 'HbA1c_level',
                            'blood_glucose_level', 'diabetes']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            print(f"错误: 数据文件缺少以下必要列: {missing_columns}")
            return None, None

        print(f"成功加载数据，共 {len(data)} 条记录")
    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        return None, None

    # 特征工程和预处理
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    smoke_map = {'never': 0, 'former': 1, 'current': 2,
                 'not current': 3, 'ever': 4, 'No Info': 5}
    data['gender'] = data['gender'].map(gender_map)
    data['smoking_history'] = data['smoking_history'].map(smoke_map)

    # 处理缺失值
    data.fillna(data.median(), inplace=True)

    # 分割特征和标签
    X = data.drop('diabetes', axis=1)
    y = data['diabetes']

    # 特征归一化
    scaler = joblib.load(scaler_path)
    scaler_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    X[scaler_columns] = scaler.transform(X[scaler_columns])

    return X, y


def test_model(model, X, y):
    """使用模型进行测试"""
    # 将pandas数据转换为numpy数组
    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(np.int32)

    # 设置模型为评估模式
    model.set_train(False)

    # 进行预测
    predictions = []
    batch_size = 128
    num_samples = X_np.shape[0]

    print("正在进行预测...")
    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        batch_X = mindspore.Tensor(X_np[i:end])

        pred = model(batch_X)
        pred_labels = (pred > 0.5).asnumpy().astype(np.int32)
        predictions.append(pred_labels)

    # 合并预测结果
    all_predictions = np.concatenate(predictions).flatten()

    return all_predictions


def evaluate_results(y_true, y_pred):
    """评估模型结果"""
    print("\n" + "=" * 40)
    print("模型评估结果")
    print("=" * 40)

    # 计算各种指标
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # 打印结果
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"F1分数 (F1 Score): {f1:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n混淆矩阵:")
    print(f"真阴性 (TN): {tn} | 假阳性 (FP): {fp}")
    print(f"假阴性 (FN): {fn} | 真阳性 (TP): {tp}")

    # 计算特异度
    specificity = tn / (tn + fp)
    print(f"特异度 (Specificity): {specificity:.4f}")

    return {
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'confusion_matrix': cm
    }


def main(args):
    # 显示数据格式信息
    print_data_format_info()

    # 检查模型文件是否存在
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return

    # 检查缩放器文件是否存在
    scaler_path = args.scaler_path
    if not os.path.exists(scaler_path):
        print(f"错误: 缩放器文件不存在: {scaler_path}")
        return

    # 预处理数据
    print(f"正在处理测试数据: {args.data_file}")
    X, y = preprocess_data(args.data_file, scaler_path)
    if X is None or y is None:
        return

    # 加载模型
    print(f"正在加载模型: {model_path}")
    model = ImprovedNetwork()
    param_dict = load_checkpoint(model_path)
    load_param_into_net(model, param_dict)

    # 测试模型
    y_pred = test_model(model, X, y)

    # 评估结果
    metrics = evaluate_results(y.values, y_pred)

    print("\n测试完成!")

    # 如果需要保存结果
    if args.save_results:
        result_file = args.save_results
        result_dir = os.path.dirname(result_file)
        if result_dir and not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # 将结果保存为CSV
        results_df = pd.DataFrame({
            'actual': y.values,
            'predicted': y_pred
        })
        results_df.to_csv(result_file, index=False)
        print(f"预测结果已保存至: {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='糖尿病预测模型测试')
    parser.add_argument('--data_file', type=str, default='dataset/diabetes_prediction_dataset.csv',
                        help='测试数据CSV文件路径')
    parser.add_argument('--model_path', type=str, default='models/best_model.ckpt',
                        help='训练好的模型文件路径')
    parser.add_argument('--scaler_path', type=str, default='models/scaler.pkl',
                        help='特征缩放器文件路径')
    parser.add_argument('--save_results', type=str, default='',
                        help='保存预测结果的CSV文件路径 (可选)')

    args = parser.parse_args()

    main(args)