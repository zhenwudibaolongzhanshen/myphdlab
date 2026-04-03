import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA   # 新增导入
import pandas as pd
import os

# -------------------- 配置参数 --------------------
FILE_PATH = r'E:\BaiduNetdiskDownload\大连理工容量预测数据\公开数据集验证\椅子图片数据集\RC-49_64x64.h5'
BATCH_SIZE = 64
EPOCHS = 30  # 增加训练轮数
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 特征提取设置
EXTRACT_NUM = 15000  # 若内存不足可限制样本数，例如50000
SAVE_CSV = False     # 建议设为False，直接保存npy

# -------------------- 1. 加载数据 --------------------
print("Loading data...")
with h5py.File(FILE_PATH, 'r') as f:
    images = f['images'][:]  # (176400, 3, 64, 64) uint8
    labels = f['labels'][:]  # (176400,) float64
    indx_train = f['indx_train'][:]  # 训练索引
    indx_valid = f['indx_valid'][:]  # 验证索引

# 归一化图像到 [0,1]
images = images.astype(np.float32) / 255.0
labels = labels.astype(np.float32)

# -------------------- 2. 重新划分数据集 --------------------
X_train_full = images[indx_train]
y_train_full = labels[indx_train]
X_test = images[indx_valid]  # 原验证集作为测试集
y_test = labels[indx_valid]

# 从原训练集中划分新的训练集和验证集（80% 训练，20% 验证）
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

print(f"新训练集: {X_train.shape}, 新验证集: {X_val.shape}, 测试集: {X_test.shape}")

# -------------------- 3. 创建 DataLoader --------------------
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -------------------- 4. 定义模型 --------------------
class RegCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc_input_dim = 128 * 8 * 8
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        features = x.view(x.size(0), -1)
        if return_features:
            return features
        x = F.relu(self.fc1(features))
        out = self.fc2(x)
        return out.squeeze()


model = RegCNN().to(DEVICE)

# -------------------- 5. 训练模型（增加早停） --------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float('inf')
patience = 5
counter = 0

print("开始训练...")
for epoch in range(EPOCHS):
    # 训练
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 早停检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))
print("最佳模型已加载。")


# -------------------- 6. 特征提取函数 --------------------
def extract_features(model, dataloader, device, max_samples=None):
    model.eval()
    features_list = []
    labels_list = []
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            feats = model(inputs, return_features=True).cpu().numpy()
            labels_list.append(targets.numpy())
            features_list.append(feats)
            total += inputs.size(0)
            if max_samples is not None and total >= max_samples:
                excess = total - max_samples
                if excess > 0:
                    features_list[-1] = features_list[-1][:-excess]
                    labels_list[-1] = labels_list[-1][:-excess]
                break
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels


# -------------------- 7. 分别提取训练集、验证集、测试集特征 --------------------
print("提取训练集特征...")
train_features, train_labels = extract_features(model, train_loader, DEVICE, max_samples=EXTRACT_NUM)
print(f"训练集特征形状: {train_features.shape}")

print("提取验证集特征...")
val_features, val_labels = extract_features(model, val_loader, DEVICE, max_samples=EXTRACT_NUM)
print(f"验证集特征形状: {val_features.shape}")

print("提取测试集特征...")
test_features, test_labels = extract_features(model, test_loader, DEVICE, max_samples=EXTRACT_NUM)
print(f"测试集特征形状: {test_features.shape}")

# -------------------- 8. PCA降维到200维（新增部分）--------------------
print("开始PCA降维到200维...")
pca = PCA(n_components=200, random_state=42, svd_solver='randomized')
train_features_pca = pca.fit_transform(train_features)
val_features_pca = pca.transform(val_features)
test_features_pca = pca.transform(test_features)

print(f"降维后训练集特征形状: {train_features_pca.shape}")
print(f"降维后验证集特征形状: {val_features_pca.shape}")
print(f"降维后测试集特征形状: {test_features_pca.shape}")

# 替换原特征变量，后续保存将使用降维后的特征
train_features = train_features_pca
val_features = val_features_pca
test_features = test_features_pca

# -------------------- 9. 保存特征（降维后）--------------------
if SAVE_CSV:
    # 保存为CSV（此时特征为200维，体积已大幅减小）
    print("正在保存训练集CSV...")
    train_df = pd.DataFrame(train_features)
    train_df['label'] = train_labels
    train_df.to_csv('train_features.csv', index=False)

    print("正在保存验证集CSV...")
    val_df = pd.DataFrame(val_features)
    val_df['label'] = val_labels
    val_df.to_csv('val_features.csv', index=False)

    print("正在保存测试集CSV...")
    test_df = pd.DataFrame(test_features)
    test_df['label'] = test_labels
    test_df.to_csv('test_features.csv', index=False)

    print("所有CSV文件保存完成。")
else:
    # 保存为numpy格式（推荐）
    np.save('train_features.npy', train_features)
    np.save('train_labels.npy', train_labels)
    np.save('val_features.npy', val_features)
    np.save('val_labels.npy', val_labels)
    np.save('test_features.npy', test_features)
    np.save('test_labels.npy', test_labels)
    print("降维后的特征已保存为numpy格式。")