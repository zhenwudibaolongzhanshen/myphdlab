# ============================================================
# CatBoost 训练脚本 - 使用 Deep Ensemble 筛选后的数据集
# 功能：加载 Deep Ensemble 筛选后的 CSV 文件，训练 CatBoost 回归模型，
#       并输出测试集上的各项指标（MSE, MAE, R², MAPE, 最大APE）。
# 参数与 MC Dropout 筛选后的 CatBoost 脚本完全一致，便于对比。
# ============================================================

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
# Deep Ensemble 筛选后数据所在目录（由 deep_ensemble_reselect.py 生成）
FILTERED_DATA_DIR = 'deep_ensemble_pca_new_ratio'

# 文件命名格式（与筛选脚本输出一致）
TRAIN_FILE = 'train_filtered.csv'
VAL_FILE = 'val_filtered.csv'
TEST_FILE = 'test_filtered.csv'

# -------------------- 1. 加载筛选后的数据 --------------------
print("加载 Deep Ensemble 筛选后的数据...")
train_df = pd.read_csv(f"{FILTERED_DATA_DIR}/{TRAIN_FILE}")
val_df = pd.read_csv(f"{FILTERED_DATA_DIR}/{VAL_FILE}")
test_df = pd.read_csv(f"{FILTERED_DATA_DIR}/{TEST_FILE}")

# 分离特征和标签（假设最后一列名为 'target'）
X_train = train_df.iloc[:, :-1].values
y_train = train_df['target'].values
X_val = val_df.iloc[:, :-1].values
y_val = val_df['target'].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df['target'].values

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

# -------------------- 2. 定义评估指标函数 --------------------
def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-6):
    """计算MAPE（百分比），避免除以零"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
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

# -------------------- 4. 设置CatBoost参数（与MC Dropout筛选后保持一致）--------------------
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.02,
    depth=4,
    loss_function='RMSE',
    eval_metric='MAE',
    random_seed=42,
    early_stopping_rounds=50,   # 早停轮数
    task_type='GPU',             # 若GPU不可用请改为'CPU'
    devices='0',
    verbose=50
)

# -------------------- 5. 训练（使用验证集早停）--------------------
print("开始训练CatBoost...")
model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True,        # 使用验证集上最好的模型
    plot=False
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

print("\n======= 测试集结果 (Deep Ensemble 筛选后) =======")
print(f"MSE  : {mse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")
print(f"最大绝对误差率 : {max_ape:.2f}%")