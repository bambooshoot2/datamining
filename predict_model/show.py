import os
import pandas as pd
import matplotlib.pyplot as plt

# ========== 创建pic文件夹 ==========
pic_dir = 'pic'
os.makedirs(pic_dir, exist_ok=True)

# 读取日志数据
df = pd.read_csv('training_log.csv')

# ================= 1. 损失曲线（训练&测试） =================
plt.figure(figsize=(8,5))
plt.plot(df['epoch'], df['train_loss'], 'o-', label='Train Loss')
plt.plot(df['epoch'], df['test_loss'], 's-', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, 'loss_curve.png'), dpi=300)
plt.close()

# ================= 2. 测试准确率&F1分数折线图 =================
plt.figure(figsize=(8,5))
plt.plot(df['epoch'], df['test_acc'], 'o-', label='Test Accuracy')
plt.plot(df['epoch'], df['test_f1'], 's-', label='Test F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Test Accuracy and F1 Score per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, 'acc_f1_curve.png'), dpi=300)
plt.close()

# ================= 3. 最佳F1分数阶梯图 =================
plt.figure(figsize=(8,5))
plt.step(df['epoch'], df['best_f1'], where='post', color='purple', label='Best F1 So Far')
plt.xlabel('Epoch')
plt.ylabel('Best F1 Score')
plt.title('Best F1 Score Evolution')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, 'best_f1_step.png'), dpi=300)
plt.close()

# ================= 4. 综合子图（损失+准确率&F1） =================
fig, axs = plt.subplots(1, 2, figsize=(14,5))
axs[0].plot(df['epoch'], df['train_loss'], '-o', label='Train Loss')
axs[0].plot(df['epoch'], df['test_loss'], '-s', label='Test Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Loss per Epoch')
axs[0].legend()
axs[0].grid(True)
axs[1].plot(df['epoch'], df['test_acc'], '-o', label='Test Acc')
axs[1].plot(df['epoch'], df['test_f1'], '-s', label='Test F1')
axs[1].plot(df['epoch'], df['best_f1'], '-^', label='Best F1')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Score')
axs[1].set_title('Accuracy & F1 per Epoch')
axs[1].legend()
axs[1].grid(True)
plt.tight_layout()
fig.savefig(os.path.join(pic_dir, 'multi_panel.png'), dpi=300)
plt.close(fig)

# ================= 5. 测试损失与F1双Y轴对比 =================
fig, ax1 = plt.subplots(figsize=(8,5))
color1 = 'tab:orange'
color2 = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Loss', color=color1)
ax1.plot(df['epoch'], df['test_loss'], marker='o', color=color1, label='Test Loss')
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
ax2.set_ylabel('F1 Score', color=color2)
ax2.plot(df['epoch'], df['test_f1'], marker='s', color=color2, label='Test F1')
ax2.tick_params(axis='y', labelcolor=color2)
plt.title('Test Loss and F1 Score vs Epoch')
fig.tight_layout()
fig.savefig(os.path.join(pic_dir, 'loss_f1_twiny.png'), dpi=300)
plt.close(fig)

print(f'All figures saved in folder: {pic_dir}')
