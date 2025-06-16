import mindspore
import numpy as np
from model import ImprovedNetwork
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

# ====== 加载模型和scaler ======
model = ImprovedNetwork()
mindspore.load_checkpoint("best_model.ckpt", net=model)
scaler = joblib.load('scaler.pkl')  # <-- 用训练时保存的scaler！

sclaer_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# ===== 映射 =====
gender_map_cn2en = {"女性": "Female", "男性": "Male", "未知": "Other"}
smoke_map_cn2en = {
    "从未吸烟": "never", "以前吸烟": "former", "经常吸烟": "current",
    "最近没有吸烟": "not current", "最近正在吸烟": "ever", "无信息": "No Info"
}
is_map = {"无": 0, "有": 1}
gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
smoke_map = {'never': 0, 'former': 1, 'current': 2, 'not current': 3, 'ever': 4, 'No Info': 5}

# ===== FastAPI接口部分 =====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    gender: str
    age: float
    hypertension: str
    heart_disease: str
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

@app.post("/predict")
def predict_api(item: PredictRequest):
    # 分类变量编码
    gender_en = gender_map_cn2en.get(item.gender, "Other")
    smoking_en = smoke_map_cn2en.get(item.smoking_history, "No Info")
    cat_features = [
        gender_map.get(gender_en, 2),
        is_map.get(item.hypertension, 0),
        is_map.get(item.heart_disease, 0),
        smoke_map.get(smoking_en, 5)
    ]
    num_vals = np.array([[item.age, item.bmi, item.HbA1c_level, item.blood_glucose_level]], dtype=np.float32)
    num_features = scaler.transform(num_vals)[0]

    features = [
        cat_features[0],                # gender
        num_features[0],                # age(norm)
        cat_features[1],                # hypertension
        cat_features[2],                # heart_disease
        cat_features[3],                # smoking_history
        num_features[1],                # bmi(norm)
        num_features[2],                # HbA1c(norm)
        num_features[3],                # glucose(norm)
    ]

    input_arr = mindspore.Tensor(np.array([features]), dtype=mindspore.float32)
    model.set_train(False)
    logits = model(input_arr)
    pred = (logits > 0.5).astype(mindspore.int32)
    val = int(pred.asnumpy())
    return {"diabetes": val}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
