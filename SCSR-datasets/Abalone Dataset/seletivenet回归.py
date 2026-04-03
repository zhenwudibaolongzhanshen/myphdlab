import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


# -------------------- 数据加载与预处理 --------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
df = pd.read_csv(url, header=None, names=column_names)

encoder = OneHotEncoder(sparse_output=False)
sex_encoded = encoder.fit_transform(df[['Sex']])
sex_df = pd.DataFrame(sex_encoded, columns=encoder.get_feature_names_out(['Sex']))
X = pd.concat([df.drop(columns=['Sex', 'Rings']), sex_df], axis=1)
y = df['Rings'].values.astype(np.float32)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
val_size = 0.2 / 0.9
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)

print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# -------------------- 模型定义 --------------------
class SelectiveNetRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], target_coverage=0.8, lambda_reg=0.5):
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


def selective_loss(pred, conf, y, target_coverage, lambda_reg):
    batch_size = pred.size(0)
    k = max(1, int(np.ceil(target_coverage * batch_size)))
    _, indices = torch.topk(conf.squeeze(), k)

    pred_accepted = pred[indices]
    y_accepted = y[indices]
    mse_accepted = torch.mean((pred_accepted - y_accepted) ** 2)

    actual_coverage = k / batch_size
    coverage_penalty = max(0, target_coverage - actual_coverage) ** 2

    loss = mse_accepted + lambda_reg * coverage_penalty
    return loss


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=30):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred, conf = model(X_batch)
            loss = selective_loss(pred, conf, y_batch, model.target_coverage, model.lambda_reg)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred, conf = model(X_batch)
                loss = selective_loss(pred, conf, y_batch, model.target_coverage, model.lambda_reg)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_state)
    return model


# -------------------- 评估函数（用于测试）--------------------
def evaluate_selective(model, loader, threshold):
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


# -------------------- 主程序 --------------------
input_dim = X_train.shape[1]
model = SelectiveNetRegressor(input_dim, hidden_dims=[64, 32],
                              target_coverage=0.8, lambda_reg=0.5)

model = train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=30)

# 保存模型
torch.save(model.state_dict(), 'selectivenet_model.pth')
print("模型已保存为 selectivenet_model.pth")

# 在验证集上计算所有样本的置信度并保存
model.eval()
all_confs_val = []
with torch.no_grad():
    for X_batch, _ in val_loader:
        _, conf = model(X_batch)
        all_confs_val.append(conf.cpu().numpy())
confs_val = np.concatenate(all_confs_val).flatten()
np.save('val_confs.npy', confs_val)
print("验证集置信度数组已保存为 val_confs.npy")

# 可选：也保存输入维度（供加载时使用）
np.save('input_dim.npy', np.array([input_dim]))
print("输入维度已保存为 input_dim.npy")