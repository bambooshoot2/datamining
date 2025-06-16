import streamlit as st
from load_model import predict

# 设置网页标题
st.title('糖尿病预测Web APP')

gender = st.selectbox('请选择患者性别', ('男性', '女性', '未知'))
age = st.number_input('请输入患者年龄', value=None, step=1, min_value=0)
hypertension = st.selectbox('请选择患者是否患有高血压', ('有', '无'))
heart_disease = st.selectbox('请选择患者是否患有心脏病', ('有', '无'))
smoking_history = st.selectbox('请选择患者吸烟史',
                               ('从未吸烟', '以前吸烟', '经常吸烟', '最近没有吸烟', '最近正在吸烟', '无信息'))
bmi = st.number_input('请输入患者BMI', value=None, step=0.01, min_value=0.00)
HbA1c_level = st.number_input('请输入患者糖化血红蛋白数值', value=None, step=0.1, min_value=0.0)
blood_glucose_level = st.number_input('请输入患者血糖数值', value=None, step=1, min_value=0)

if st.button('提交检测'):
    data = [gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]
    print(data)
    if '' not in data and None not in data:
        gender_map = {'男性': 0,
                      '女性': 1,
                      '未知': 2}
        smoke_map = {'从未吸烟': 0,
                     '无信息': 1,
                     '最近正在吸烟': 2,
                     '以前吸烟r': 3,
                     '经常吸烟': 4,
                     '最近没有吸烟': 5}
        is_map = {'无': 0,
                  '有': 1}
        data[0] = gender_map.get(data[0])
        data[1] = min((80 - data[1]) / (80 - 0.08), 1)
        data[2] = is_map.get(data[2])
        data[3] = is_map.get(data[3])
        # data[2]=1 if data
        data[4] = smoke_map.get(data[4])
        data[5] = min((91.82 - data[5]) / (91.82 - 10.01), 1)
        data[6] = min((9 - data[6]) / (9 - 3.5), 1)
        data[7] = min((300 - data[7]) / (300 - 80), 1)
        result = '有' if predict(data, 'one') == 1 else '没有'
        st.success(f'预测结果显示，该患者{result}糖尿病')
    else:
        st.info('请确认输入信息的完整性')
