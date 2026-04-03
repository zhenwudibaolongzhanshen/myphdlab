# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor

# ==================== MAPE 和 最大 APE 计算函数 ====================
def mean_absolute_percentage_error(y_true, y_pred):
    """计算 MAPE（百分比形式）"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def max_absolute_percentage_error(y_true, y_pred):
    """计算最大绝对百分比误差（百分比形式）"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.max(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ==================== 1. 加载筛选后的数据集 ====================
# 修改此处为 Deep Ensemble 的输出目录
DATA_DIR = 'deep_ensemble_filtered_data'  # 原为 'mc_dropout_filtered_data_pytorch'

train_df = pd.read_csv(f'{DATA_DIR}/train_filtered.csv')
val_df = pd.read_csv(f'{DATA_DIR}/val_filtered.csv')
test_df = pd.read_csv(f'{DATA_DIR}/test_filtered.csv')

# 分离特征和目标列（假设所有特征列名与原始一致，目标列为 'target'）
feature_names = [col for col in train_df.columns if col != 'target']
X_train = train_df[feature_names].values
y_train = train_df['target'].values
X_val = val_df[feature_names].values
y_val = val_df['target'].values
X_test = test_df[feature_names].values
y_test = test_df['target'].values

print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")
print(f"测试集大小: {len(X_test)}")
print(f"特征数: {X_train.shape[1]}")

# ==================== 2. 创建 CatBoost 回归模型 ====================
model = CatBoostRegressor(
    iterations=8000,           # 最大迭代次数
    learning_rate=0.05,         # 学习率
    depth=6,                    # 树深度
    loss_function='RMSE',       # 损失函数
    eval_metric='R2',           # 评估指标（用于早停监控）
    random_seed=42,             # 保证可复现
    verbose=100,                # 每100轮输出一次日志
    early_stopping_rounds=100   # 早停
)

# ==================== 3. 训练模型，并传入验证集用于早停 ====================
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True,
    plot=False
)

# ==================== 4. 在三个集上评估模型 ====================
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

# 计算各项指标
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

train_mae = mean_absolute_error(y_train, train_pred)
val_mae = mean_absolute_error(y_val, val_pred)
test_mae = mean_absolute_error(y_test, test_pred)

train_r2 = r2_score(y_train, train_pred)
val_r2 = r2_score(y_val, val_pred)
test_r2 = r2_score(y_test, test_pred)

train_mape = mean_absolute_percentage_error(y_train, train_pred)
val_mape = mean_absolute_percentage_error(y_val, val_pred)
test_mape = mean_absolute_percentage_error(y_test, test_pred)

train_max_ape = max_absolute_percentage_error(y_train, train_pred)
val_max_ape = max_absolute_percentage_error(y_val, val_pred)
test_max_ape = max_absolute_percentage_error(y_test, test_pred)

# 打印结果
print("\n===== 模型评估结果 =====")
print(f"{'指标':<15} {'训练集':<15} {'验证集':<15} {'测试集':<15}")
print("-" * 80)
print(f"{'RMSE':<15} {train_rmse:<15.4f} {val_rmse:<15.4f} {test_rmse:<15.4f}")
print(f"{'MAE':<15} {train_mae:<15.4f} {val_mae:<15.4f} {test_mae:<15.4f}")
print(f"{'R²':<15} {train_r2:<15.4f} {val_r2:<15.4f} {test_r2:<15.4f}")
print(f"{'MAPE(%)':<15} {train_mape:<15.2f} {val_mape:<15.2f} {test_mape:<15.2f}")
print(f"{'Max APE(%)':<15} {train_max_ape:<15.2f} {val_max_ape:<15.2f} {test_max_ape:<15.2f}")

print(f"\n实际迭代次数: {model.tree_count_}")