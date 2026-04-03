# ============================================================
# MC Dropout 样本筛选模块 (PyTorch 实现) - PCA 特征数据集版
# 功能：基于 MC Dropout 不确定性对训练/验证/测试集分别筛选 28.89% 样本
#       保留预测方差最小的样本（即模型最确定的样本）
# 数据：从 .npy 文件加载已降维的 PCA 特征
# 输出：筛选后的数据集 CSV 文件（特征 + 标签），并保存模型和标准化器
# ============================================================

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib  # 用于保存 scaler
import json
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以保证可重复性
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==================== 配置 ====================
OUTPUT_DIR = 'mc_dropout_filtered_pca'
RATIO = 0.3493
MC_ITER = 100                       # MC Dropout 迭代次数
HIDDEN_LAYERS = [100, 50, 25]       # 隐藏层神经元数
DROPOUT_RATE = 0.5
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"输出目录: {OUTPUT_DIR}")
print(f"使用设备: {DEVICE}")

# ==================== 1. 加载 PCA 降维后的数据 ====================
print("\n" + "=" * 60)
print("加载 PCA 特征数据")
print("=" * 60)

X_train = np.load('train_features.npy')
y_train = np.load('train_labels.npy')
X_val = np.load('val_features.npy')
y_val = np.load('val_labels.npy')
X_test = np.load('test_features.npy')
y_test = np.load('test_labels.npy')

# 转换为 DataFrame（便于筛选和保存）
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

# ==================== 2. 数据标准化（用训练集拟合）====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df.values)
X_val_scaled = scaler.transform(X_val_df.values)
X_test_scaled = scaler.transform(X_test_df.values)

# 保存标准化器，便于后续预测时使用
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
print("标准化器已保存。")

# 转换为 PyTorch 张量
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# ==================== 3. 定义带 Dropout 的 MLP 模型 ====================
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
        layers.append(nn.Linear(prev_dim, 1))  # 回归输出
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def enable_dropout(self):
        """将 Dropout 层设置为训练模式（即使整体处于 eval 模式也启用）"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

input_dim = X_train_t.shape[1]
model = MLPWithDropout(input_dim, HIDDEN_LAYERS, DROPOUT_RATE).to(DEVICE)
print(model)

# ==================== 4. 训练模型（仅使用训练集）====================
print("\n" + "=" * 60)
print("训练 MC Dropout 基础模型")
print("=" * 60)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
    avg_loss = total_loss / len(train_dataset)
    if (epoch+1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

print("基础模型训练完成。")

# ==================== 5. 保存模型和配置 ====================
# 保存模型权重
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pth'))
# 保存模型配置，便于重新实例化
model_config = {
    'input_dim': input_dim,
    'hidden_layers': HIDDEN_LAYERS,
    'dropout_rate': DROPOUT_RATE
}
with open(os.path.join(OUTPUT_DIR, 'model_config.json'), 'w') as f:
    json.dump(model_config, f, indent=2)
print("模型权重和配置已保存。")

# ==================== 6. MC Dropout 预测函数 ====================
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

# 将数据移到设备
X_train_t_device = X_train_t.to(DEVICE)
X_val_t_device = X_val_t.to(DEVICE)
X_test_t_device = X_test_t.to(DEVICE)

# ==================== 7. 分别计算三个数据集的不确定性 ====================
print("\n" + "=" * 60)
print("计算 MC Dropout 不确定性")
print("=" * 60)

train_std = mc_dropout_std(model, X_train_t_device)
val_std = mc_dropout_std(model, X_val_t_device)
test_std = mc_dropout_std(model, X_test_t_device)

print("不确定性计算完成。")

# ==================== 8. 根据不确定性筛选样本（保留方差最小的 RATIO 比例）====================
def select_low_std_indices(std_values, ratio=RATIO):
    n = len(std_values)
    k = int(n * ratio)
    # 按标准差升序排序，取前 k 个
    indices = np.argsort(std_values)[:k]
    return indices

train_idx = select_low_std_indices(train_std)
val_idx = select_low_std_indices(val_std)
test_idx = select_low_std_indices(test_std)

X_train_selected = X_train_df.iloc[train_idx].values
y_train_selected = y_train.iloc[train_idx].values
X_val_selected = X_val_df.iloc[val_idx].values
y_val_selected = y_val.iloc[val_idx].values
X_test_selected = X_test_df.iloc[test_idx].values
y_test_selected = y_test.iloc[test_idx].values

print("\n筛选后样本数:")
print(f"训练集: {len(X_train_selected)}")
print(f"验证集: {len(X_val_selected)}")
print(f"测试集: {len(X_test_selected)}")

# ==================== 9. 保存筛选后的数据集为 CSV ====================
def save_csv(X, y, name):
    # X 是 numpy 数组，使用 feature_names 作为列名
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    path = os.path.join(OUTPUT_DIR, f'{name}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"已保存: {path}")

save_csv(X_train_selected, y_train_selected, 'train_filtered')
save_csv(X_val_selected, y_val_selected, 'val_filtered')
save_csv(X_test_selected, y_test_selected, 'test_filtered')

# ==================== 10. 保存筛选索引（可选）====================
np.savez(os.path.join(OUTPUT_DIR, 'selected_indices.npz'),
         train=train_idx, val=val_idx, test=test_idx)
print("筛选索引已保存。")

# ==================== 11. 保存配置信息 ====================
config = {
    'description': 'MC Dropout based sample selection (keep low-uncertainty samples) - PCA features',
    'retained_ratio': RATIO,
    'mc_iterations': MC_ITER,
    'model_hidden_layers': HIDDEN_LAYERS,
    'dropout_rate': DROPOUT_RATE,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
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
print("配置信息已保存。")

print("\n" + "=" * 60)
print("MC Dropout 样本筛选完成！")
print(f"所有输出保存在: {OUTPUT_DIR}")
print("=" * 60)