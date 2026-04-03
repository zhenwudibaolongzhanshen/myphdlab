# ============================================================
# catboost_abalone_with_ecoc_filter.py
# 功能：在鲍鱼数据集（含Sex独热编码）上，使用ECOC软分类概率筛选样本，
#       然后训练CatBoost回归模型，评估并保存结果。
# 依赖：需先运行 ECOC-MLP 训练脚本（带Sex列），生成模型文件：
#       ecoc_abalone_with_sex_output/ecoc_mlp_abalone.pkl
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 添加ECOCMLPTrainer类的定义（与训练时完全一致）====================
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def assign_segment_labels(y_values, boundaries):
    """根据边界给连续值分配分段标签(0~5)"""
    n_segments = len(boundaries) - 1
    labels = np.zeros(len(y_values), dtype=int)
    for i, val in enumerate(y_values):
        if val <= boundaries[0]:
            labels[i] = 0
        elif val >= boundaries[-1]:
            labels[i] = n_segments - 1
        else:
            for seg in range(n_segments):
                if boundaries[seg] <= val <= boundaries[seg+1]:
                    labels[i] = seg
                    break
    return labels

class ECOCMLPTrainer:
    def __init__(self, code_matrix, boundaries, n_segments, feature_names, random_state=42):
        self.code_matrix = code_matrix
        self.boundaries = boundaries
        self.n_segments = n_segments
        self.feature_names = feature_names
        self.code_length = code_matrix.shape[1]
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.binary_classifiers = []      # 每个二分类器的模型
        self.binary_accuracies = []        # 在验证集上的准确率
        self.train_history = {}             # 可存储训练信息

    def fit(self, X_train, y_train, X_val, y_val):
        """训练所有二分类器（此方法在加载模型时不会调用，但保留以确保类完整性）"""
        pass

    def predict_proba(self, X):
        """返回软分类概率 (n_samples, n_segments)"""
        X_scaled = self.scaler.transform(X)
        n_samples = X_scaled.shape[0]

        # 收集所有二分类器的概率
        binary_probs = np.zeros((n_samples, self.code_length))
        for col, clf in enumerate(self.binary_classifiers):
            if clf is None:
                binary_probs[:, col] = 0.5
            else:
                binary_probs[:, col] = clf.predict_proba(X_scaled)[:, 1]

        # 解码得到各分段距离，再转换为概率（温度参数0.1）
        distances = np.zeros((n_samples, self.n_segments))
        for s in range(n_samples):
            for c in range(self.n_segments):
                total_dist = 0.0
                valid_bits = 0
                for b in range(self.code_length):
                    bit = self.code_matrix[c, b]
                    if bit == 0:
                        continue
                    prob = binary_probs[s, b]
                    if bit == 1:
                        total_dist += 1.0 - prob
                    else:  # bit == -1
                        total_dist += prob
                    valid_bits += 1
                distances[s, c] = total_dist / valid_bits if valid_bits > 0 else self.code_length

        probs = np.zeros((n_samples, self.n_segments))
        for s in range(n_samples):
            exp_neg = np.exp(-distances[s] / 0.1)
            probs[s] = exp_neg / np.sum(exp_neg)
        return probs

    def predict(self, X):
        """硬分类预测（取概率最大的分段）"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def save(self, path):
        """保存模型（此方法在加载时不会调用）"""
        pass

# ==================== 指标计算函数 ====================
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

# ==================== 筛选条件函数 ====================
def condition1_top3_continuous(prob, n_segments):
    """条件1：top3概率对应的分段是否连续"""
    top3_idx = np.argsort(prob)[-3:][::-1]
    sorted_top3 = np.sort(top3_idx)
    return np.all(np.diff(sorted_top3) == 1)

def condition4_top1_gt(prob, threshold=0.7):
    """条件4：最高概率大于阈值"""
    return np.max(prob) > threshold

def filter_condition_14(prob, n_segments):
    """方案：条件1 + 条件4"""
    return condition1_top3_continuous(prob, n_segments) and condition4_top1_gt(prob)

# 方案列表
SCHEMES = [
    ('14', lambda prob, n: filter_condition_14(prob, n), '条件1+4'),
    ('none', None, '无筛选（全部数据）')

]

# ==================== 训练并评估CatBoost（单次） ====================
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                       feature_names, random_state, scheme_desc, output_dir):
    print(f"\n{'=' * 50}")
    print(f"训练方案：{scheme_desc}")
    print(f"{'=' * 50}")

    # 检测GPU
    use_gpu = False
    try:
        test_model = CatBoostRegressor(iterations=1, task_type='GPU', devices='0',
                                       verbose=False, allow_writing_files=False)
        test_model.fit(X_train.iloc[:10], y_train.iloc[:10])
        use_gpu = True
        print("   GPU可用，将使用GPU加速")
    except:
        print("   GPU不可用，使用CPU训练")

    # 创建模型
    model = CatBoostRegressor(
        iterations=8000,
        learning_rate=0.01,
        depth=10,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=random_state,
        verbose=100,
        early_stopping_rounds=150,
        task_type='GPU' if use_gpu else 'CPU',
        devices='0' if use_gpu else None,
        l2_leaf_reg=200,
        min_data_in_leaf=200,
        bootstrap_type='Bayesian',
        bagging_temperature=1,
        random_strength=1,
        grow_policy='Depthwise',
        use_best_model=True
    )

    # 训练
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        plot=False
    )

    # 预测
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # 计算指标
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # MAPE 和 最大 APE
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
    train_max_ape = max_absolute_percentage_error(y_train, y_train_pred)
    val_mape = mean_absolute_percentage_error(y_val, y_val_pred)
    val_max_ape = max_absolute_percentage_error(y_val, y_val_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    test_max_ape = max_absolute_percentage_error(y_test, y_test_pred)

    # 打印结果
    print(f"\n模型评估结果:")
    print(f"{'指标':<15} {'训练集':<15} {'验证集':<15} {'测试集':<15}")
    print("-" * 80)
    print(f"{'RMSE':<15} {train_rmse:<15.4f} {val_rmse:<15.4f} {test_rmse:<15.4f}")
    print(f"{'MAE':<15} {train_mae:<15.4f} {val_mae:<15.4f} {test_mae:<15.4f}")
    print(f"{'R²':<15} {train_r2:<15.4f} {val_r2:<15.4f} {test_r2:<15.4f}")
    print(f"{'MAPE(%)':<15} {train_mape:<15.2f} {val_mape:<15.2f} {test_mape:<15.2f}")
    print(f"{'Max APE(%)':<15} {train_max_ape:<15.2f} {val_max_ape:<15.2f} {test_max_ape:<15.2f}")

    # 组织结果字典
    results = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'y_train_pred': y_train_pred, 'y_val_pred': y_val_pred, 'y_test_pred': y_test_pred,
        'train_rmse': train_rmse, 'train_mae': train_mae, 'train_r2': train_r2,
        'val_rmse': val_rmse, 'val_mae': val_mae, 'val_r2': val_r2,
        'test_rmse': test_rmse, 'test_mae': test_mae, 'test_r2': test_r2,
        'train_mape': train_mape, 'train_max_ape': train_max_ape,
        'val_mape': val_mape, 'val_max_ape': val_max_ape,
        'test_mape': test_mape, 'test_max_ape': test_max_ape,
        'model': model
    }

    save_results(results, feature_names, output_dir, scheme_desc)
    return results

# ==================== 保存结果（含可视化） ====================
def save_results(results, feature_names, output_dir, scheme_desc):
    os.makedirs(output_dir, exist_ok=True)
    model = results['model']

    # 1. 保存模型
    model_path = os.path.join(output_dir, 'catboost_model.cbm')
    model.save_model(model_path)
    print(f"\n   模型已保存: {model_path}")

    # 2. 保存预测结果（测试集）
    pred_df = pd.DataFrame({
        'y_test': results['y_test'].values,
        'y_test_pred': results['y_test_pred']
    })
    pred_path = os.path.join(output_dir, 'predictions.csv')
    pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
    print(f"   预测结果已保存: {pred_path}")

    # 3. 保存评估指标
    metrics_df = pd.DataFrame({
        '指标': ['RMSE', 'MAE', 'R²', 'MAPE(%)', '最大APE(%)'],
        '训练集': [results['train_rmse'], results['train_mae'], results['train_r2'],
                   results['train_mape'], results['train_max_ape']],
        '验证集': [results['val_rmse'], results['val_mae'], results['val_r2'],
                   results['val_mape'], results['val_max_ape']],
        '测试集': [results['test_rmse'], results['test_mae'], results['test_r2'],
                   results['test_mape'], results['test_max_ape']]
    })
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"   评估指标已保存: {metrics_path}")

    # 4. 保存特征重要性
    importance = model.get_feature_importance()
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    imp_path = os.path.join(output_dir, 'catboost_feature_importance.csv')
    imp_df.to_csv(imp_path, index=False, encoding='utf-8-sig')
    print(f"   特征重要性已保存: {imp_path}")

    # 5. 保存使用的特征列表
    features_df = pd.DataFrame({
        'feature': feature_names,
        'rank': range(1, len(feature_names) + 1)
    })
    features_path = os.path.join(output_dir, 'used_features.csv')
    features_df.to_csv(features_path, index=False, encoding='utf-8-sig')
    print(f"   使用的特征列表已保存: {features_path}")

    # 6. 保存模型参数
    params_path = os.path.join(output_dir, 'model_parameters.txt')
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write(f"CatBoost模型参数 - {scheme_desc}\n")
        f.write("=" * 50 + "\n")
        f.write(f"使用的特征数量: {len(feature_names)}\n")
        f.write(f"随机种子: 42\n")
        f.write(f"数据分割比例: 训练集70%, 验证集20%, 测试集10%\n")
        f.write(f"\n模型评估结果:\n")
        f.write(f"训练集 RMSE: {results['train_rmse']:.4f}\n")
        f.write(f"训练集 MAE: {results['train_mae']:.4f}\n")
        f.write(f"训练集 R²: {results['train_r2']:.4f}\n")
        f.write(f"训练集 MAPE: {results['train_mape']:.2f}%\n")
        f.write(f"训练集 最大APE: {results['train_max_ape']:.2f}%\n")
        f.write(f"验证集 RMSE: {results['val_rmse']:.4f}\n")
        f.write(f"验证集 MAE: {results['val_mae']:.4f}\n")
        f.write(f"验证集 R²: {results['val_r2']:.4f}\n")
        f.write(f"验证集 MAPE: {results['val_mape']:.2f}%\n")
        f.write(f"验证集 最大APE: {results['val_max_ape']:.2f}%\n")
        f.write(f"测试集 RMSE: {results['test_rmse']:.4f}\n")
        f.write(f"测试集 MAE: {results['test_mae']:.4f}\n")
        f.write(f"测试集 R²: {results['test_r2']:.4f}\n")
        f.write(f"测试集 MAPE: {results['test_mape']:.2f}%\n")
        f.write(f"测试集 最大APE: {results['test_max_ape']:.2f}%\n")
    print(f"   模型参数已保存: {params_path}")

    # 7. 可视化
    visualize_results(results, feature_names, output_dir, scheme_desc)

def visualize_results(results, feature_names, output_dir, scheme_desc):
    y_train = results['y_train']
    y_val = results['y_val']
    y_test = results['y_test']
    y_train_pred = results['y_train_pred']
    y_val_pred = results['y_val_pred']
    y_test_pred = results['y_test_pred']

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 训练集
    ax = axes[0, 0]
    ax.scatter(y_train, y_train_pred, alpha=0.5, s=10)
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    ax.set_xlabel('真实值')
    ax.set_ylabel('预测值')
    ax.set_title(f'训练集: R²={results["train_r2"]:.4f}, RMSE={results["train_rmse"]:.4f}')
    ax.grid(alpha=0.3)

    # 验证集
    ax = axes[0, 1]
    ax.scatter(y_val, y_val_pred, alpha=0.5, s=10, color='orange')
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    ax.set_xlabel('真实值')
    ax.set_ylabel('预测值')
    ax.set_title(f'验证集: R²={results["val_r2"]:.4f}, RMSE={results["val_rmse"]:.4f}')
    ax.grid(alpha=0.3)

    # 测试集
    ax = axes[0, 2]
    ax.scatter(y_test, y_test_pred, alpha=0.5, s=10, color='green')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('真实值')
    ax.set_ylabel('预测值')
    ax.set_title(f'测试集: R²={results["test_r2"]:.4f}, RMSE={results["test_rmse"]:.4f}')
    ax.grid(alpha=0.3)

    # 残差图
    # 训练集
    ax = axes[1, 0]
    residuals = y_train - y_train_pred
    ax.scatter(y_train_pred, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('预测值')
    ax.set_ylabel('残差')
    ax.set_title('训练集残差')
    ax.grid(alpha=0.3)

    # 验证集
    ax = axes[1, 1]
    residuals = y_val - y_val_pred
    ax.scatter(y_val_pred, residuals, alpha=0.5, s=10, color='orange')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('预测值')
    ax.set_ylabel('残差')
    ax.set_title('验证集残差')
    ax.grid(alpha=0.3)

    # 测试集
    ax = axes[1, 2]
    residuals = y_test - y_test_pred
    ax.scatter(y_test_pred, residuals, alpha=0.5, s=10, color='green')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('预测值')
    ax.set_ylabel('残差')
    ax.set_title('测试集残差')
    ax.grid(alpha=0.3)

    plt.suptitle(f'CatBoost 回归结果 - {scheme_desc}')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   可视化图表已保存: {save_path}")

    # 特征重要性图
    importance = results['model'].get_feature_importance()
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    imp_df = imp_df.sort_values('importance', ascending=True)
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(imp_df)), imp_df['importance'].values)
    plt.yticks(range(len(imp_df)), imp_df['feature'].values)
    plt.xlabel('特征重要性')
    plt.title(f'CatBoost特征重要性 - {scheme_desc}')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   特征重要性图已保存: {save_path}")

# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("ECOC筛选 + CatBoost回归 - 鲍鱼数据集 (保留Sex列并独热编码)")
    print("=" * 80)

    # 加载ECOC模型（带Sex列版本）
    ecoc_model_path = r'ecoc_abalone_with_sex_output/ecoc_mlp_abalone.pkl'
    if not os.path.exists(ecoc_model_path):
        raise FileNotFoundError(f"请先训练ECOC模型（带Sex列）: {ecoc_model_path}")
    trainer = joblib.load(ecoc_model_path)
    n_segments = trainer.n_segments
    print(f"ECOC模型加载成功，分段数: {n_segments}")

    # 加载鲍鱼数据集
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                    'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
    df = pd.read_csv(url, header=None, names=column_names)

    # 分离特征和目标，并对Sex进行独热编码（与ECOC训练时完全一致）
    X_df = df.drop('Rings', axis=1)
    y = df['Rings']

    # 独热编码
    X_encoded = pd.get_dummies(X_df, columns=['Sex'], prefix=['Sex'])

    # 特征名称（顺序由 get_dummies 自动生成，通常按字母顺序：Sex_F, Sex_I, Sex_M）
    feature_names = X_encoded.columns.tolist()
    print(f"\n特征名称: {feature_names}")

    X = X_encoded.values
    y = y.values

    # 划分数据集（与ECOC训练时完全一致：70%训练，20%验证，10%测试，随机种子42）
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42
    )

    # 转换为DataFrame，方便后续处理
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_val = pd.DataFrame(X_val, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    y_train = pd.Series(y_train, name='target')
    y_val = pd.Series(y_val, name='target')
    y_test = pd.Series(y_test, name='target')

    print(f"\n训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")

    # 计算每个样本的概率（软分类）
    print("\n计算ECOC软分类概率...")
    probs_train = trainer.predict_proba(X_train.values)
    probs_val = trainer.predict_proba(X_val.values)
    probs_test = trainer.predict_proba(X_test.values)

    # 输出目录基础
    output_base_dir = 'catboost_filtered_abalone_with_sex'

    # 遍历方案
    for scheme_id, filter_func, scheme_desc in SCHEMES:
        print("\n" + "=" * 70)
        print(f"当前方案：{scheme_desc} (ID: {scheme_id})")
        print("=" * 70)

        if filter_func is not None:
            train_mask = np.array([filter_func(p, n_segments) for p in probs_train])
            val_mask = np.array([filter_func(p, n_segments) for p in probs_val])
            test_mask = np.array([filter_func(p, n_segments) for p in probs_test])

            print(f"\n   筛选后：")
            print(f"     训练集保留: {np.sum(train_mask)} / {len(train_mask)} ({np.mean(train_mask) * 100:.2f}%)")
            print(f"     验证集保留: {np.sum(val_mask)} / {len(val_mask)} ({np.mean(val_mask) * 100:.2f}%)")
            print(f"     测试集保留: {np.sum(test_mask)} / {len(test_mask)} ({np.mean(test_mask) * 100:.2f}%)")

            X_train_filt = X_train[train_mask]
            y_train_filt = y_train[train_mask]
            X_val_filt = X_val[val_mask]
            y_val_filt = y_val[val_mask]
            X_test_filt = X_test[test_mask]
            y_test_filt = y_test[test_mask]
        else:
            X_train_filt = X_train
            y_train_filt = y_train
            X_val_filt = X_val
            y_val_filt = y_val
            X_test_filt = X_test
            y_test_filt = y_test
            print(f"\n   无筛选，使用全部数据")

        # 检查是否为空
        if len(X_train_filt) == 0 or len(X_val_filt) == 0 or len(X_test_filt) == 0:
            print(f"\n   ⚠ 筛选后训练集、验证集或测试集为空，跳过该方案")
            continue

        # 输出目录
        suffix = '_none' if filter_func is None else f'_{scheme_id}'
        output_dir = output_base_dir + suffix

        # 训练并评估
        train_and_evaluate(
            X_train_filt, y_train_filt,
            X_val_filt, y_val_filt,
            X_test_filt, y_test_filt,
            feature_names,
            random_state=42,
            scheme_desc=scheme_desc,
            output_dir=output_dir
        )

        print(f"\n   方案 {scheme_desc} 完成，结果保存在: {output_dir}")

    print("\n" + "=" * 70)
    print("所有方案运行完毕！")
    print("=" * 70)

if __name__ == "__main__":
    main()