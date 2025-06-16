import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('diabetes_infer_result.csv')
y_true = df['real_diabetes']
y_pred = df['pred_diabetes']

print('混淆矩阵：')
print(confusion_matrix(y_true, y_pred))
print()
print('分类指标：')
print(classification_report(y_true, y_pred, digits=4))
