import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# ==================== 新增：MAPE 和 最大 APE 计算函数 ====================
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

# 1. 加载鲍鱼数据集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
df = pd.read_csv(url, header=None, names=column_names)

# 查看数据基本信息
print("数据集形状:", df.shape)
print("\n前5行数据:")
print(df.head())
print("\n数据类型:")
print(df.dtypes)
print("\n目标变量Rings的统计描述:")
print(df['Rings'].describe())

# 2. 划分数据集（70%训练，20%验证，10%测试）
X = df.drop('Rings', axis=1)
y = df['Rings']

# 先分出测试集（占原始数据的10%）
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 从剩余的训练+验证集中分出验证集（验证集占原始数据的20%）
# 此时 X_train_val 占原始数据的 90%，验证集应占这部分的比例为 20%/90% ≈ 0.2222
val_size = 0.2 / 0.9  # 约 0.222222...
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)

# 打印划分结果确认
print(f"\n训练集样本数: {X_train.shape[0]}")
print(f"验证集样本数: {X_val.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# 3. 定义CatBoost回归模型（使用较保守的参数防止过拟合）
model = CatBoostRegressor(
    iterations=300,           # 减少迭代次数
    learning_rate=0.05,       # 较低的学习率
    depth=4,                   # 限制树深度
    l2_leaf_reg=3,             # 增加L2正则化
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=50,                # 每50轮输出一次
    early_stopping_rounds=30   # 早停防止过拟合
)

# 4. 训练模型（指定性别列为分类特征）
categorical_features = ['Sex']  # 指定性别为分类特征

model.fit(
    X_train, y_train,
    cat_features=categorical_features,
    eval_set=(X_val, y_val),
    use_best_model=True,
    plot=False
)

# 5. 在训练集、验证集、测试集上评估
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

# 计算各项指标
# RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

# MAE
train_mae = mean_absolute_error(y_train, train_pred)
val_mae = mean_absolute_error(y_val, val_pred)
test_mae = mean_absolute_error(y_test, test_pred)

# R²
train_r2 = r2_score(y_train, train_pred)
val_r2 = r2_score(y_val, val_pred)
test_r2 = r2_score(y_test, test_pred)

# MAPE
train_mape = mean_absolute_percentage_error(y_train, train_pred)
val_mape = mean_absolute_percentage_error(y_val, val_pred)
test_mape = mean_absolute_percentage_error(y_test, test_pred)

# Max APE
train_max_ape = max_absolute_percentage_error(y_train, train_pred)
val_max_ape = max_absolute_percentage_error(y_val, val_pred)
test_max_ape = max_absolute_percentage_error(y_test, test_pred)

# 打印结果表格
print("\n===== 模型评估结果 =====")
print(f"{'指标':<15} {'训练集':<15} {'验证集':<15} {'测试集':<15}")
print("-" * 80)
print(f"{'RMSE':<15} {train_rmse:<15.4f} {val_rmse:<15.4f} {test_rmse:<15.4f}")
print(f"{'MAE':<15} {train_mae:<15.4f} {val_mae:<15.4f} {test_mae:<15.4f}")
print(f"{'R²':<15} {train_r2:<15.4f} {val_r2:<15.4f} {test_r2:<15.4f}")
print(f"{'MAPE(%)':<15} {train_mape:<15.2f} {val_mape:<15.2f} {test_mape:<15.2f}")
print(f"{'Max APE(%)':<15} {train_max_ape:<15.2f} {val_max_ape:<15.2f} {test_max_ape:<15.2f}")

# 输出实际迭代次数
print(f"\n实际迭代次数: {model.tree_count_}")



