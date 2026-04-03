import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# -------------------- 1. 加载PCA降维后的特征和标签 --------------------
print("加载数据...")
X_train = np.load('train_features.npy')
y_train = np.load('train_labels.npy')
X_val = np.load('val_features.npy')
y_val = np.load('val_labels.npy')
X_test = np.load('test_features.npy')
y_test = np.load('test_labels.npy')

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

# -------------------- 2. 定义评估指标函数 --------------------
def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-6):
    """计算MAPE（百分比），避免除以零"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 只计算真实值非零的样本（或添加极小值）
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def max_absolute_percentage_error(y_true, y_pred, epsilon=1e-6):
    """计算最大绝对百分比误差（百分比）"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return np.nan
    return np.max(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# -------------------- 3. 创建CatBoost数据集（支持验证集）--------------------
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)
test_pool = Pool(X_test, y_test)

# -------------------- 4. 设置CatBoost参数 --------------------
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.02,
    depth=4,
    loss_function='RMSE',
    eval_metric='MAE',
    random_seed=42,
    early_stopping_rounds=50,   # 早停轮数，会自动启用迭代早停
    task_type='GPU',             # 若GPU不可用改为'CPU'
    devices='0',
    verbose=50
)

# -------------------- 5. 训练（使用验证集早停）--------------------
print("开始训练CatBoost...")
model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True,               # 使用验证集上最好的模型
    plot=False                          # 如需绘图可设为True（需安装ipywidgets）
)

# -------------------- 6. 预测与评估 --------------------
print("\n在测试集上评估...")
y_pred = model.predict(X_test)

# 计算各项指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
max_ape = max_absolute_percentage_error(y_test, y_pred)

print("\n======= 测试集结果 =======")
print(f"MSE  : {mse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")
print(f"最大绝对误差率 : {max_ape:.2f}%")