import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取CSV文件
csv_path = "/root/hq_anomaly/hq_anomaly/output/0719_tinyxianshuban/roc_curve_metric.csv"
df = pd.read_csv(csv_path)

# 提取三列数据
fpr = df["fpr"].values
tpr = df["tpr"].values
thr = df["thr"].values

# 2. 计算AUC（梯形积分）
def calculate_auc(x, y):
    auc = 0.0
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        avg_y = (y[i] + y[i-1]) / 2
        auc += dx * avg_y
    return auc
auroc = calculate_auc(fpr, tpr)

# 3. 寻找零漏检（TPR≈1）对应的最小FPR
mask = tpr >= 0.9999
if mask.any():
    min_fpr_zero_miss = fpr[mask].min()
else:
    min_fpr_zero_miss = None

# 4. 绘图
plt.figure(figsize=(7, 7), dpi=120)
# ROC曲线
plt.plot(fpr, tpr, color="#1f77b4", linewidth=1.5, label=f"ROC curve (AUROC={auroc:.4f})")
# 随机猜测对角线
plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random Guess")

plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
plt.xlabel("FPR (False Positive Rate / 误检率)")
plt.ylabel("TPR (True Positive Rate / 缺陷检出率)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# 保存图片
plt.savefig("roc_plot.png", bbox_inches="tight")
plt.show()

# 打印关键指标
print(f"AUROC: {auroc:.4f}")
if min_fpr_zero_miss is not None:
    print(f"漏检率=0时最小误检率FPR: {min_fpr_zero_miss:.4f}")
else:
    print("不存在能100%检出所有缺陷的阈值")