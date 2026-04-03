import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# -------------------- 指标函数 --------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def max_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.max(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# -------------------- 模型定义（与训练时一致）--------------------
class SelectiveNetRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], target_coverage=0.6, lambda_reg=0.2):
        super().__init__()
        self.target_coverage = target_coverage
        self.lambda_reg = lambda_reg

        layers = []
        dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        self.pred_head = nn.Linear(dim, 1)
        self.conf_head = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.shared(x)
        pred = self.pred_head(features)
        conf = self.conf_head(features)
        return pred, conf


# -------------------- 评估函数 --------------------
def evaluate_selective(model, loader, threshold):
    """返回选择性预测的指标字典（包括覆盖率）"""
    model.eval()
    all_preds = []
    all_confs = []
    all_y = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            pred, conf = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_confs.append(conf.cpu().numpy())
            all_y.append(y_batch.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0).flatten()
    confs = np.concatenate(all_confs, axis=0).flatten()
    y_true = np.concatenate(all_y, axis=0).flatten()

    accept_mask = confs >= threshold
    y_accepted = y_true[accept_mask]
    pred_accepted = preds[accept_mask]
    coverage = np.mean(accept_mask)

    if len(y_accepted) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan,
                'MAPE': np.nan, 'Max APE': np.nan, 'Coverage': coverage}

    rmse = np.sqrt(mean_squared_error(y_accepted, pred_accepted))
    mae = mean_absolute_error(y_accepted, pred_accepted)
    r2 = r2_score(y_accepted, pred_accepted)
    mape = mean_absolute_percentage_error(y_accepted, pred_accepted)
    max_ape = max_absolute_percentage_error(y_accepted, pred_accepted)

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2,
            'MAPE': mape, 'Max APE': max_ape, 'Coverage': coverage}


def evaluate_all(model, loader):
    """全样本评估（无拒绝）"""
    model.eval()
    all_preds = []
    all_y = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            pred, _ = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_y.append(y_batch.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0).flatten()
    y_true = np.concatenate(all_y, axis=0).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    mape = mean_absolute_percentage_error(y_true, preds)
    max_ape = max_absolute_percentage_error(y_true, preds)

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2,
            'MAPE': mape, 'Max APE': max_ape}


# -------------------- 数据加载与预处理（与训练时相同）--------------------
data = load_diabetes()
X, y = data.data, data.target

# 划分训练集 (70%)、验证集 (20%)、测试集 (10%) —— 与训练时保持一致
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42
)

# 加载训练时保存的标准化器并转换测试集
scaler = torch.load('scaler_diabetes.pth', weights_only=False)
X_test_scaled = scaler.transform(X_test)

X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

batch_size = 32
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# -------------------- 加载模型和置信度 --------------------
input_dim = np.load('input_dim_diabetes.npy')[0].item()
model = SelectiveNetRegressor(input_dim, hidden_dims=[128, 64, 32])  # 超参数应与训练一致
model.load_state_dict(torch.load('selectivenet_diabetes_model.pth', map_location='cpu'))
model.eval()
print("模型加载完成")

confs_val = np.load('val_confs_diabetes.npy')
print(f"验证集置信度统计: 最小值={confs_val.min():.4f}, 最大值={confs_val.max():.4f}, "
      f"均值={confs_val.mean():.4f}, 中位数={np.median(confs_val):.4f}")


def threshold_from_coverage(confs, target_coverage):
    """根据置信度数组和目标覆盖率返回阈值"""
    if target_coverage <= 0:
        return 1.1  # 全拒绝
    if target_coverage >= 1:
        return -0.1  # 全接受
    return np.percentile(confs, (1 - target_coverage) * 100)


# -------------------- 多覆盖率遍历 --------------------
coverages_to_try = np.linspace(0.1, 0.9, 9)  # 从 10% 到 90% 共 9 个点
results = []

for target_cov in coverages_to_try:
    thresh = threshold_from_coverage(confs_val, target_cov)
    metrics = evaluate_selective(model, test_loader, thresh)
    metrics['Target_Coverage'] = target_cov
    results.append(metrics)

# 添加全样本点（覆盖率=1）
all_metrics = evaluate_all(model, test_loader)
all_metrics['Coverage'] = 1.0
all_metrics['Target_Coverage'] = 1.0
results.append(all_metrics)

df_results = pd.DataFrame(results)
print("\n各目标覆盖率下的测试集性能：")
print(df_results.round(4))

# -------------------- 创建保存文件夹 --------------------
save_dir = "精度图_diabetes_fixed"
os.makedirs(save_dir, exist_ok=True)

# 保存数据到 CSV
csv_path = os.path.join(save_dir, "coverage_metrics.csv")
df_results.to_csv(csv_path, index=False)
print(f"\n数据已保存至: {csv_path}")

# -------------------- 绘制四个指标曲线 --------------------
metrics_info = {
    'RMSE': {'title': 'Coverage-RMSE Curve (Diabetes - Fixed)', 'ylabel': 'RMSE (Lower is Better)', 'better': 'lower'},
    'MAE': {'title': 'Coverage-MAE Curve (Diabetes - Fixed)', 'ylabel': 'MAE (Lower is Better)', 'better': 'lower'},
    'R2': {'title': 'Coverage-R² Curve (Diabetes - Fixed)', 'ylabel': 'R² (Higher is Better)', 'better': 'higher'},
    'MAPE': {'title': 'Coverage-MAPE Curve (Diabetes - Fixed)', 'ylabel': 'MAPE (%) (Lower is Better)', 'better': 'lower'}
}

for metric, info in metrics_info.items():
    plt.figure(figsize=(8, 5))
    mask_select = df_results['Coverage'] < 1.0
    plt.plot(df_results.loc[mask_select, 'Coverage'],
             df_results.loc[mask_select, metric],
             marker='o', linestyle='-', label='Selective Prediction')
    full_point = df_results[df_results['Coverage'] == 1.0]
    if not full_point.empty:
        plt.scatter(full_point['Coverage'], full_point[metric],
                    color='red', s=100, zorder=5,
                    label=f"Full (Coverage=1.0, {metric}={full_point[metric].iloc[0]:.3f})")
    plt.xlabel('Coverage (Proportion of Accepted Samples)')
    plt.ylabel(info['ylabel'])
    plt.title(info['title'])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"coverage_{metric.lower()}.png")
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"图片已保存至: {fig_path}")

print("\n所有曲线生成完毕。")