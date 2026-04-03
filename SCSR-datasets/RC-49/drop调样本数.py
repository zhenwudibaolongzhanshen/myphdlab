# ============================================================
# 加载已训练模型，重新进行 MC Dropout 样本筛选（调整比例）
# 功能：加载之前保存的模型权重、配置和标准化器，对 PCA 特征数据
#       重新进行 MC Dropout 不确定性计算，并按新的比例筛选样本。
# 输入：原始 PCA 特征文件（train_features.npy 等）和模型文件目录
# 输出：按新比例筛选后的 CSV 文件及索引
# ============================================================

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
# 之前保存模型的目录（包含 model.pth, model_config.json, scaler.pkl）
MODEL_DIR = 'mc_dropout_filtered_pca'

# 新筛选的输出目录
OUTPUT_DIR = 'mc_dropout_filtered_pca_new_ratio'

# 新的筛选比例（可在此修改）
NEW_RATIO = 0.3928  # 例如改为保留 50% 的样本

# MC Dropout 迭代次数（应与训练时一致，或可调整）
MC_ITER = 100

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 随机种子（保证可重复性）
np.random.seed(42)
torch.manual_seed(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"模型目录: {MODEL_DIR}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"新筛选比例: {NEW_RATIO}")
print(f"使用设备: {DEVICE}")

# ==================== 1. 加载 PCA 原始数据 ====================
print("\n" + "=" * 60)
print("加载 PCA 特征数据")
print("=" * 60)

X_train = np.load('train_features.npy')
y_train = np.load('train_labels.npy')
X_val = np.load('val_features.npy')
y_val = np.load('val_labels.npy')
X_test = np.load('test_features.npy')
y_test = np.load('test_labels.npy')

# 构建 DataFrame 保留特征名（便于保存 CSV）
feature_names = [f'PC{i+1}' for i in range(X_train.shape[1])]
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_val_df = pd.DataFrame(X_val, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)
y_train = pd.Series(y_train, name='target')
y_val = pd.Series(y_val, name='target')
y_test = pd.Series(y_test, name='target')

print(f"训练集: {len(X_train_df)} 样本, 特征维度: {X_train_df.shape[1]}")
print(f"验证集: {len(X_val_df)} 样本")
print(f"测试集: {len(X_test_df)} 样本")

# ==================== 2. 加载标准化器并标准化数据 ====================
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
scaler = joblib.load(scaler_path)
print("标准化器加载成功。")

X_train_scaled = scaler.transform(X_train_df.values)
X_val_scaled = scaler.transform(X_val_df.values)
X_test_scaled = scaler.transform(X_test_df.values)

# 转换为 PyTorch 张量
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)

# ==================== 3. 重建模型并加载权重 ====================
# 加载模型配置
config_path = os.path.join(MODEL_DIR, 'model_config.json')
if not os.path.exists(config_path):
    raise FileNotFoundError(f"模型配置文件不存在: {config_path}")
with open(config_path, 'r') as f:
    model_config = json.load(f)

input_dim = model_config['input_dim']
hidden_layers = model_config['hidden_layers']
dropout_rate = model_config['dropout_rate']

# 定义模型结构（与训练时相同）
class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def enable_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

# 实例化模型并加载权重
model = MLPWithDropout(input_dim, hidden_layers, dropout_rate).to(DEVICE)
model_path = os.path.join(MODEL_DIR, 'model.pth')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型权重文件不存在: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()  # 切换到评估模式（但后续会手动启用 Dropout）
print("模型加载成功。")

# ==================== 4. MC Dropout 预测函数 ====================
def mc_dropout_std(model, X, n_iter=MC_ITER):
    """
    对输入 X 进行 MC Dropout 预测，返回每个样本的预测标准差
    model: 已训练的 PyTorch 模型
    X: 输入张量 (n_samples, n_features)，已在对应设备上
    """
    model.eval()
    model.enable_dropout()   # 强制开启 Dropout
    predictions = []
    with torch.no_grad():
        for _ in range(n_iter):
            pred = model(X).cpu().numpy().flatten()
            predictions.append(pred)
    predictions = np.array(predictions)  # (n_iter, n_samples)
    std = predictions.std(axis=0)
    return std

# ==================== 5. 计算不确定性 ====================
print("\n" + "=" * 60)
print("计算 MC Dropout 不确定性")
print("=" * 60)

train_std = mc_dropout_std(model, X_train_t)
val_std = mc_dropout_std(model, X_val_t)
test_std = mc_dropout_std(model, X_test_t)

print("不确定性计算完成。")

# ==================== 6. 根据新比例筛选样本 ====================
def select_low_std_indices(std_values, ratio):
    n = len(std_values)
    k = int(n * ratio)
    indices = np.argsort(std_values)[:k]
    return indices

train_idx = select_low_std_indices(train_std, NEW_RATIO)
val_idx = select_low_std_indices(val_std, NEW_RATIO)
test_idx = select_low_std_indices(test_std, NEW_RATIO)

# 提取筛选后的数据（使用原始未标准化的 DataFrame，以便保存原始特征值）
X_train_selected = X_train_df.iloc[train_idx].values
y_train_selected = y_train.iloc[train_idx].values
X_val_selected = X_val_df.iloc[val_idx].values
y_val_selected = y_val.iloc[val_idx].values
X_test_selected = X_test_df.iloc[test_idx].values
y_test_selected = y_test.iloc[test_idx].values

print("\n新比例筛选后样本数:")
print(f"训练集: {len(X_train_selected)}")
print(f"验证集: {len(X_val_selected)}")
print(f"测试集: {len(X_test_selected)}")

# ==================== 7. 保存筛选后的数据集 ====================
def save_csv(X, y, name):
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    path = os.path.join(OUTPUT_DIR, f'{name}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"已保存: {path}")

save_csv(X_train_selected, y_train_selected, 'train_filtered')
save_csv(X_val_selected, y_val_selected, 'val_filtered')
save_csv(X_test_selected, y_test_selected, 'test_filtered')

# 保存筛选索引
np.savez(os.path.join(OUTPUT_DIR, 'selected_indices.npz'),
         train=train_idx, val=val_idx, test=test_idx)
print("筛选索引已保存。")

# ==================== 8. 保存新配置信息 ====================
config = {
    'description': 'MC Dropout sample selection with new ratio (loaded pretrained model)',
    'model_source_dir': MODEL_DIR,
    'new_retained_ratio': NEW_RATIO,
    'mc_iterations': MC_ITER,
    'device': str(DEVICE),
    'original_train_size': len(X_train_df),
    'selected_train_size': len(X_train_selected),
    'original_val_size': len(X_val_df),
    'selected_val_size': len(X_val_selected),
    'original_test_size': len(X_test_df),
    'selected_test_size': len(X_test_selected)
}
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print("新配置信息已保存。")

print("\n" + "=" * 60)
print("按新比例筛选完成！")
print(f"输出保存在: {OUTPUT_DIR}")
print("=" * 60)