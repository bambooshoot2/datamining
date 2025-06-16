import pandas as pd
import mindspore
import numpy as np
from model import ImprovedNetwork
import joblib

# === 类别相关映射表 ===
gender_map_cn2en = {"女性": "Female", "男性": "Male", "未知": "Other"}
smoke_map_cn2en = {
    "从未吸烟": "never", "以前吸烟": "former", "经常吸烟": "current",
    "最近没有吸烟": "not current", "最近正在吸烟": "ever", "无信息": "No Info"
}
is_map = {"无": 0, "有": 1}
gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
smoke_map = {'never': 0, 'former': 1, 'current': 2, 'not current': 3, 'ever': 4, 'No Info': 5}
gender_en2zh = {v: k for k, v in gender_map_cn2en.items()}
smoke_en2zh = {v: k for k, v in smoke_map_cn2en.items()}
yn_map = {0: '无', 1: '有'}

sclaer_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

def batch_form_from_csv(df):
    """英文csv转为中文表单批量DataFrame"""
    batch_df = pd.DataFrame({
        "gender": df["gender"].map(gender_en2zh).fillna("未知"),
        "age": df["age"].astype(float),
        "hypertension": df["hypertension"].map(yn_map).fillna("无"),
        "heart_disease": df["heart_disease"].map(yn_map).fillna("无"),
        "smoking_history": df["smoking_history"].map(smoke_en2zh).fillna("无信息"),
        "bmi": df["bmi"].astype(float),
        "HbA1c_level": df["HbA1c_level"].astype(float),
        "blood_glucose_level": df["blood_glucose_level"].astype(float)
    })
    return batch_df

def batch_feature_matrix(form_df, scaler):
    """批量表单DataFrame转标准特征(N,8)，数值特征用scaler归一化"""
    gender_en = form_df['gender'].map(gender_map_cn2en).fillna("Other")
    smoking_en = form_df['smoking_history'].map(smoke_map_cn2en).fillna("No Info")

    # 仅对这四列用训练的scaler归一化
    num_features = scaler.transform(form_df[sclaer_columns])

    feat_mat = np.stack([
        gender_en.map(gender_map).fillna(2).astype(int),     # gender
        num_features[:, 0],                                  # age(norm)
        form_df['hypertension'].map(is_map).fillna(0).astype(int),  # hypertension
        form_df['heart_disease'].map(is_map).fillna(0).astype(int), # heart_disease
        smoking_en.map(smoke_map).fillna(5).astype(int),     # smoking_history
        num_features[:, 1],                                  # bmi(norm)
        num_features[:, 2],                                  # HbA1c_level(norm)
        num_features[:, 3],                                  # blood_glucose_level(norm)
    ], axis=1)
    return feat_mat.astype(np.float32)

if __name__ == '__main__':
    # ===== 加载模型和scaler =====
    model = ImprovedNetwork()
    mindspore.load_checkpoint("best_model.ckpt", net=model)
    scaler = joblib.load('scaler.pkl')  # ← 载入训练时scaler参数！

    # ===== 读csv到DataFrame =====
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    form_df = batch_form_from_csv(df)
    feat_mat = batch_feature_matrix(form_df, scaler)

    # ===== 一次性批量推理！=====
    model.set_train(False)
    input_tensor = mindspore.Tensor(feat_mat, dtype=mindspore.float32)
    logits = model(input_tensor)
    logits_np = logits.asnumpy().squeeze()
    pred = (logits_np > 0.5).astype(int)
    score = logits_np

    out_df = form_df.copy()
    out_df['real_diabetes'] = df['diabetes'].astype(int)
    out_df['pred_diabetes'] = pred
    out_df['score'] = score
    out_df.to_csv('diabetes_infer_result.csv', index=False, encoding='utf_8_sig')
    print('批量推理已完成，已保存到 diabetes_infer_result.csv')