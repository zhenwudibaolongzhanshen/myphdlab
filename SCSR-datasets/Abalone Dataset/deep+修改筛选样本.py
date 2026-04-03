# ============================================================
# Deep Ensemble 样本筛选比例调整脚本（无需重新训练）
# 功能：基于已保存的 Deep Ensemble 模型和 scaler，
#       重新计算训练/验证/测试集的不确定性（标准差），
#       并按新的保留比例筛选样本，输出覆盖原文件。
# 使用前提：已运行过 `deep_ensemble_abalone.py`，且模型文件保存在
#          `deep_ensemble_abalone` 目录下。
# ============================================================

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
# 请在此处修改你想要的新保留比例（例如 0.3 表示保留 30% 的样本）
NEW_RATIO = 0.3493

# 原始输出目录（必须与训练脚本中的 OUTPUT_DIR 一致）
MODEL_DIR = 'deep_ensemble_abalone'

# 设备自动检测
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# ==================== 1. 加载保存的模型、scaler 和特征名称 ====================
print("=" * 60)
print("加载已保存的 Deep Ensemble 模型及附属文件")
print("=" * 60)

# 从配置文件读取集成数量和隐藏层结构
with open(os.path.join(MODEL_DIR, 'config.json'), 'r') as f:
    config = json.load(f)
N_ENSEMBLE = config['ensemble_size']
HIDDEN_LAYERS = config['model_hidden_layers']
print(f"集成模型数: {N_ENSEMBLE}")
print(f"隐藏层结构: {HIDDEN_LAYERS}")

# 重新定义模型结构（必须与训练时完全一致）
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            prev_dim = units
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 加载模型
loaded_models = []
for i in range(N_ENSEMBLE):
    model_path = os.path.join(MODEL_DIR, f'model_{i}.pth')
    # 输入维度需要从 scaler 的 n_features_in_ 获得，但这里先加载一个临时模型以获取维度
    # 更好的做法是从 feature_names.txt 获取特征数量
    with open(os.path.join(MODEL_DIR, 'feature_names.txt'), 'r') as f:
        feature_names = [line.strip() for line in f]
    input_dim = len(feature_names)

    model = MLP(input_dim, HIDDEN_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    loaded_models.append(model)
print(f"成功加载 {len(loaded_models)} 个模型。")

# 加载标准化器
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
print("标准化器 (scaler) 加载成功。")

# 加载特征名称（用于构建输出 DataFrame）
print(f"特征名称: {feature_names}")

# ==================== 2. 重新加载原始鲍鱼数据集并预处理 ====================
print("\n" + "=" * 60)
print("重新加载鲍鱼数据集并进行预处理")
print("=" * 60)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
df = pd.read_csv(url, header=None, names=column_names)

# 分离特征和目标
X_df = df.drop('Rings', axis=1)
y = df['Rings'].values

# 对 Sex 进行独热编码（必须与训练时一致：pd.get_dummies 默认按字母顺序生成三列）
X_encoded = pd.get_dummies(X_df, columns=['Sex'], prefix=['Sex'])
X = X_encoded.values

# 划分数据集（必须使用与训练时完全相同的 random_state，以保证划分一致）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42
)

print(f"原始训练集: {len(X_train)} 样本")
print(f"原始验证集: {len(X_val)} 样本")
print(f"原始测试集: {len(X_test)} 样本")

# 对数据应用之前拟合的 scaler 进行标准化
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 转为 PyTorch 张量
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)

# ==================== 3. 定义集成标准差计算函数 ====================
def ensemble_std(models, X):
    """返回每个样本的集成预测标准差"""
    predictions = []
    for model in models:
        with torch.no_grad():
            pred = model(X).cpu().numpy().flatten()
        predictions.append(pred)
    predictions = np.array(predictions)  # (n_models, n_samples)
    return predictions.std(axis=0)

# ==================== 4. 重新计算三个数据集的不确定性 ====================
print("\n" + "=" * 60)
print(f"使用新比例 {NEW_RATIO:.2f} 重新计算不确定性并筛选")
print("=" * 60)

train_std = ensemble_std(loaded_models, X_train_t)
val_std = ensemble_std(loaded_models, X_val_t)
test_std = ensemble_std(loaded_models, X_test_t)

# ==================== 5. 按新比例选取索引 ====================
def select_low_std_indices(std_values, ratio):
    n = len(std_values)
    k = int(n * ratio)
    indices = np.argsort(std_values)[:k]   # 标准差升序，取前 k 个
    return indices

train_idx = select_low_std_indices(train_std, NEW_RATIO)
val_idx = select_low_std_indices(val_std, NEW_RATIO)
test_idx = select_low_std_indices(test_std, NEW_RATIO)

# 从原始（未标准化）数据中提取子集
X_train_selected = X_train[train_idx]
y_train_selected = y_train[train_idx]
X_val_selected = X_val[val_idx]
y_val_selected = y_val[val_idx]
X_test_selected = X_test[test_idx]
y_test_selected = y_test[test_idx]

print("\n新比例筛选后样本数:")
print(f"训练集: {len(X_train_selected)} / {len(X_train)} ({len(X_train_selected)/len(X_train)*100:.2f}%)")
print(f"验证集: {len(X_val_selected)} / {len(X_val)} ({len(X_val_selected)/len(X_val)*100:.2f}%)")
print(f"测试集: {len(X_test_selected)} / {len(X_test)} ({len(X_test_selected)/len(X_test)*100:.2f}%)")

# ==================== 6. 保存筛选后的数据集（覆盖原文件）====================
print("\n" + "=" * 60)
print("保存新筛选后的 CSV 文件（将覆盖原文件）")
print("=" * 60)

def save_csv(X, y, name):
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    path = os.path.join(MODEL_DIR, f'{name}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"已覆盖: {path}")

save_csv(X_train_selected, y_train_selected, 'train_filtered')
save_csv(X_val_selected, y_val_selected, 'val_filtered')
save_csv(X_test_selected, y_test_selected, 'test_filtered')

# 同时保存新的筛选索引（可选，文件名添加比例以示区别）
np.savez(os.path.join(MODEL_DIR, f'selected_indices_ratio{NEW_RATIO}.npz'),
         train=train_idx, val=val_idx, test=test_idx)
print(f"新筛选索引已保存至: {os.path.join(MODEL_DIR, f'selected_indices_ratio{NEW_RATIO}.npz')}")

# 可选：更新配置文件中的保留比例（但不覆盖原配置，可另存为新文件）
config['retained_ratio'] = NEW_RATIO
config['selected_train_size'] = len(X_train_selected)
config['selected_val_size'] = len(X_val_selected)
config['selected_test_size'] = len(X_test_selected)
with open(os.path.join(MODEL_DIR, f'config_ratio{NEW_RATIO}.json'), 'w') as f:
    json.dump(config, f, indent=2)
print(f"新配置已保存至: {os.path.join(MODEL_DIR, f'config_ratio{NEW_RATIO}.json')}")

print("\n" + "=" * 60)
print(f"筛选比例调整为 {NEW_RATIO:.2f} 完成！")
print("=" * 60)