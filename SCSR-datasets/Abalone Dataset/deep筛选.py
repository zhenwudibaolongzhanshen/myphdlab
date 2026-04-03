# ============================================================
# Deep Ensemble 样本筛选模块 (PyTorch 实现) - 鲍鱼数据集版
# 功能：基于 Deep Ensemble (5个独立MLP) 的不确定性对训练/验证/测试集分别筛选指定比例样本
#       保留预测方差最小的样本（即模型集成最确定的样本）
# 数据：UCI Abalone，对 Sex 列进行独热编码，按 70/20/10 划分
# 输出：筛选后的数据集 CSV 文件（特征 + 标签），并保存模型和 scaler 以供后续灵活调整比例
# ============================================================

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以保证可重复性（每个模型会使用自己的种子，但总体可复现）
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==================== 配置 ====================
OUTPUT_DIR = 'deep_ensemble_abalone'
RATIO = 0.8942                    # 保留比例 50% (可后续通过加载模型重新计算调整)
N_ENSEMBLE = 5                  # 集成模型数量
HIDDEN_LAYERS = [100, 50, 25]   # 隐藏层神经元数
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"输出目录: {OUTPUT_DIR}")
print(f"使用设备: {DEVICE}")
print(f"集成模型数: {N_ENSEMBLE}")

# ==================== 1. 加载并预处理鲍鱼数据集 ====================
print("=" * 60)
print("加载鲍鱼数据集（对Sex列进行独热编码）")
print("=" * 60)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
df = pd.read_csv(url, header=None, names=column_names)

# 分离特征和目标
X_df = df.drop('Rings', axis=1)
y = df['Rings'].values

# 对 Sex 进行独热编码（生成 Sex_F, Sex_I, Sex_M 三列）
X_encoded = pd.get_dummies(X_df, columns=['Sex'], prefix=['Sex'])
feature_names = X_encoded.columns.tolist()   # 保存特征名，用于输出CSV
print(f"特征数量: {len(feature_names)}")
print(f"特征名: {feature_names}")

X = X_encoded.values

# 划分训练(70%)、验证(20%)、测试(10%)（与 MC Dropout 和 CatBoost 代码保持一致）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42
)

print(f"原始训练集: {len(X_train)} 样本")
print(f"原始验证集: {len(X_val)} 样本")
print(f"原始测试集: {len(X_test)} 样本")

# ==================== 2. 数据标准化（用训练集拟合）====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 转换为 PyTorch 张量
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ==================== 3. 定义 MLP 模型（无 Dropout）====================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            # 可选的 BatchNorm，但为了简单此处不加
            prev_dim = units
        layers.append(nn.Linear(prev_dim, 1))  # 回归输出
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

input_dim = X_train_t.shape[1]

# ==================== 4. 训练 N_ENSEMBLE 个独立模型 ====================
def train_model(seed):
    """使用指定随机种子训练一个 MLP 模型"""
    torch.manual_seed(seed)
    model = MLP(input_dim, HIDDEN_LAYERS).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 创建 DataLoader
    train_dataset = TensorDataset(X_train_t.to(DEVICE), y_train_t.to(DEVICE))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
        # 可选打印
        # if (epoch+1) % 50 == 0:
        #     avg_loss = total_loss / len(train_dataset)
        #     print(f"Model {seed} Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    return model

print("\n" + "=" * 60)
print(f"开始训练 {N_ENSEMBLE} 个独立 MLP 模型")
print("=" * 60)
models = []
for i in range(N_ENSEMBLE):
    seed = 42 + i  # 不同的随机种子
    print(f"训练模型 {i+1}/{N_ENSEMBLE} (seed={seed})...")
    model = train_model(seed)
    models.append(model)

    # 保存每个模型的参数
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_{i}.pth'))

print("所有模型训练完成，模型参数已保存。")

# 保存特征名称（用于后续构建 DataFrame）
with open(os.path.join(OUTPUT_DIR, 'feature_names.txt'), 'w') as f:
    f.write('\n'.join(feature_names))

# 保存标准化器
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
print("标准化器 (scaler) 已保存。")

# ==================== 5. Deep Ensemble 预测函数 ====================
def ensemble_std(models, X):
    """
    对输入 X 进行集成预测，返回每个样本的预测标准差
    models: 模型列表
    X: 输入张量 (n_samples, n_features)，已在对应设备上
    """
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(X).cpu().numpy().flatten()
        predictions.append(pred)
    predictions = np.array(predictions)  # (n_models, n_samples)
    std = predictions.std(axis=0)
    return std

# 将数据移到设备
X_train_t_device = X_train_t.to(DEVICE)
X_val_t_device = X_val_t.to(DEVICE)
X_test_t_device = X_test_t.to(DEVICE)

# ==================== 6. 分别计算三个数据集的不确定性 ====================
print("\n" + "=" * 60)
print("计算 Deep Ensemble 不确定性 (标准差)")
print("=" * 60)

train_std = ensemble_std(models, X_train_t_device)
val_std = ensemble_std(models, X_val_t_device)
test_std = ensemble_std(models, X_test_t_device)

print("不确定性计算完成。")

# ==================== 7. 根据不确定性筛选样本（保留方差最小的 RATIO 比例）====================
def select_low_std_indices(std_values, ratio=RATIO):
    n = len(std_values)
    k = int(n * ratio)
    indices = np.argsort(std_values)[:k]  # 升序，取前 k 个
    return indices

train_idx = select_low_std_indices(train_std)
val_idx = select_low_std_indices(val_std)
test_idx = select_low_std_indices(test_std)

X_train_selected = X_train[train_idx]
y_train_selected = y_train[train_idx]
X_val_selected = X_val[val_idx]
y_val_selected = y_val[val_idx]
X_test_selected = X_test[test_idx]
y_test_selected = y_test[test_idx]

print("\n筛选后样本数:")
print(f"训练集: {len(X_train_selected)} / {len(X_train)} ({len(X_train_selected)/len(X_train)*100:.2f}%)")
print(f"验证集: {len(X_val_selected)} / {len(X_val)} ({len(X_val_selected)/len(X_val)*100:.2f}%)")
print(f"测试集: {len(X_test_selected)} / {len(X_test)} ({len(X_test_selected)/len(X_test)*100:.2f}%)")

# ==================== 8. 保存筛选后的数据集为 CSV ====================
def save_csv(X, y, name):
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    path = os.path.join(OUTPUT_DIR, f'{name}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"已保存: {path}")

save_csv(X_train_selected, y_train_selected, 'train_filtered')
save_csv(X_val_selected, y_val_selected, 'val_filtered')
save_csv(X_test_selected, y_test_selected, 'test_filtered')

# ==================== 9. 保存筛选索引（可选）====================
np.savez(os.path.join(OUTPUT_DIR, 'selected_indices.npz'),
         train=train_idx, val=val_idx, test=test_idx)
print("筛选索引已保存。")

# ==================== 10. 保存配置信息 ====================
config = {
    'description': 'Deep Ensemble based sample selection (keep low-uncertainty samples) - Abalone dataset',
    'retained_ratio': RATIO,
    'ensemble_size': N_ENSEMBLE,
    'model_hidden_layers': HIDDEN_LAYERS,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'device': str(DEVICE),
    'original_train_size': len(X_train),
    'selected_train_size': len(X_train_selected),
    'original_val_size': len(X_val),
    'selected_val_size': len(X_val_selected),
    'original_test_size': len(X_test),
    'selected_test_size': len(X_test_selected),
    'feature_names': feature_names
}

with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print("配置信息已保存。")

print("\n" + "=" * 60)
print("Deep Ensemble 样本筛选完成！")
print(f"所有输出保存在: {OUTPUT_DIR}")
print("=" * 60)