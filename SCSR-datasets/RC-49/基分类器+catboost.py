# ============================================================
# ecoc_filtered_catboost_multi.py
# 功能：分别加载三种基分类器（逻辑回归/CatBoost/随机森林）训练得到的ECOC模型，
#       利用软分类概率筛选样本，然后训练CatBoost回归模型，评估并保存结果。
# 依赖：需先运行 ecoc_base_models_after_pca.py，生成模型文件：
#       ecoc_logistic_output/ecoc_logistic_pca.pkl
#       ecoc_catboost_output/ecoc_catboost_pca.pkl
#       ecoc_randomforest_output/ecoc_randomforest_pca.pkl
# 输入：train_features.npy, train_labels.npy,
#       val_features.npy,   val_labels.npy,
#       test_features.npy,  test_labels.npy
# 输出目录：catboost_filtered_pca_<基分类器>_<方案>/
# ============================================================

import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 筛选条件函数（与原代码一致）====================
def condition1_top3_continuous(prob, n_segments):
    """条件1：top3概率对应的分段是否连续"""
    top3_idx = np.argsort(prob)[-3:][::-1]
    sorted_top3 = np.sort(top3_idx)
    return np.all(np.diff(sorted_top3) == 1)

def condition4_top1_gt(prob, threshold=0.85):
    """条件4：最高概率大于阈值"""
    return np.max(prob) > threshold

def filter_condition_14(prob, n_segments):
    """方案：条件1 + 条件4"""
    return condition1_top3_continuous(prob, n_segments) and condition4_top1_gt(prob)

# 方案列表（保留无筛选作为对照）
SCHEMES = [
    ('none', None, '无筛选（全部数据）'),
    ('14', lambda prob, n: filter_condition_14(prob, n), '条件1+4')
]

# ==================== 指标计算函数 ====================
def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-6):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def max_absolute_percentage_error(y_true, y_pred, epsilon=1e-6):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return np.nan
    return np.max(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ==================== 训练并评估CatBoost（单次）====================
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                       feature_names, random_state, scheme_desc, output_dir, model_tag):
    print(f"\n{'=' * 50}")
    print(f"基分类器: {model_tag} | 方案：{scheme_desc}")
    print(f"{'=' * 50}")

    # 检测GPU
    use_gpu = False
    try:
        test_model = CatBoostRegressor(iterations=1, task_type='GPU', devices='0',
                                       verbose=False, allow_writing_files=False)
        test_model.fit(X_train[:10], y_train[:10])
        use_gpu = True
        print("   GPU可用，将使用GPU加速")
    except:
        print("   GPU不可用，使用CPU训练")

    # 创建模型
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.02,
        depth=4,
        loss_function='RMSE',
        eval_metric='MAE',
        random_seed=random_state,
        early_stopping_rounds=50,
        task_type='GPU' if use_gpu else 'CPU',
        devices='0' if use_gpu else None,
        verbose=50,
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

    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
    train_max_ape = max_absolute_percentage_error(y_train, y_train_pred)
    val_mape = mean_absolute_percentage_error(y_val, y_val_pred)
    val_max_ape = max_absolute_percentage_error(y_val, y_val_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    test_max_ape = max_absolute_percentage_error(y_test, y_test_pred)

    print(f"\n模型评估结果:")
    print(f"{'指标':<15} {'训练集':<15} {'验证集':<15} {'测试集':<15}")
    print("-" * 80)
    print(f"{'RMSE':<15} {train_rmse:<15.4f} {val_rmse:<15.4f} {test_rmse:<15.4f}")
    print(f"{'MAE':<15} {train_mae:<15.4f} {val_mae:<15.4f} {test_mae:<15.4f}")
    print(f"{'R²':<15} {train_r2:<15.4f} {val_r2:<15.4f} {test_r2:<15.4f}")
    print(f"{'MAPE(%)':<15} {train_mape:<15.2f} {val_mape:<15.2f} {test_mape:<15.2f}")
    print(f"{'Max APE(%)':<15} {train_max_ape:<15.2f} {val_max_ape:<15.2f} {test_max_ape:<15.2f}")

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

    save_results(results, feature_names, output_dir, scheme_desc, model_tag)
    return results

# ==================== 保存结果（含可视化）====================
def save_results(results, feature_names, output_dir, scheme_desc, model_tag):
    os.makedirs(output_dir, exist_ok=True)
    model = results['model']

    # 1. 保存模型
    model_path = os.path.join(output_dir, 'catboost_model.cbm')
    model.save_model(model_path)
    print(f"\n   模型已保存: {model_path}")

    # 2. 保存预测结果（测试集）
    pred_df = pd.DataFrame({
        'y_test': results['y_test'],
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
        f.write(f"CatBoost模型参数 - 基分类器: {model_tag} | {scheme_desc}\n")
        f.write("=" * 50 + "\n")
        f.write(f"使用的特征数量: {len(feature_names)}\n")
        f.write(f"随机种子: 42\n")
        f.write(f"数据分割比例: 训练集70%, 验证集20%, 测试集10% (PCA后)\n")
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
    visualize_results(results, feature_names, output_dir, scheme_desc, model_tag)

def visualize_results(results, feature_names, output_dir, scheme_desc, model_tag):
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

    plt.suptitle(f'CatBoost 回归结果 - 基分类器: {model_tag} | {scheme_desc} (PCA后)')
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
    plt.title(f'CatBoost特征重要性 - 基分类器: {model_tag} | {scheme_desc} (PCA后)')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   特征重要性图已保存: {save_path}")

# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("多基分类器ECOC筛选 + CatBoost回归 (PCA后) - 糖尿病数据集")
    print("=" * 80)

    # 定义要加载的基分类器模型路径及对应的标签
    model_configs = [
        {'name': 'logistic',   'path': 'ecoc_logistic_output/ecoc_logistic_pca.pkl'},
        {'name': 'catboost',   'path': 'ecoc_catboost_output/ecoc_catboost_pca.pkl'},
        {'name': 'randomforest','path': 'ecoc_randomforest_output/ecoc_randomforest_pca.pkl'}
    ]

    # 加载PCA降维后的数据
    print("\n加载PCA特征数据...")
    X_train = np.load('train_features.npy')
    y_train = np.load('train_labels.npy')
    X_val = np.load('val_features.npy')
    y_val = np.load('val_labels.npy')
    X_test = np.load('test_features.npy')
    y_test = np.load('test_labels.npy')

    # 转换为DataFrame（便于筛选和保存）
    feature_names = [f'PC{i+1}' for i in range(X_train.shape[1])]
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_val = pd.DataFrame(X_val, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    y_train = pd.Series(y_train, name='target')
    y_val = pd.Series(y_val, name='target')
    y_test = pd.Series(y_test, name='target')

    print(f"\n训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")

    # 遍历每个基分类器模型
    for config in model_configs:
        model_name = config['name']
        model_path = config['path']

        print("\n" + "=" * 70)
        print(f"处理基分类器: {model_name}")
        print("=" * 70)

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"   ⚠ 模型文件不存在: {model_path}，跳过")
            continue

        # 加载ECOC模型
        try:
            trainer = joblib.load(model_path)
            n_segments = trainer.n_segments
            print(f"   ECOC模型加载成功，分段数: {n_segments}")
        except Exception as e:
            print(f"   ⚠ 加载模型失败: {e}，跳过")
            continue

        # 计算每个样本的软分类概率
        print("\n   计算ECOC软分类概率...")
        probs_train = trainer.predict_proba(X_train.values)
        probs_val = trainer.predict_proba(X_val.values)
        probs_test = trainer.predict_proba(X_test.values)

        # 遍历筛选方案
        for scheme_id, filter_func, scheme_desc in SCHEMES:
            print(f"\n   当前方案：{scheme_desc} (ID: {scheme_id})")

            if filter_func is not None:
                train_mask = np.array([filter_func(p, n_segments) for p in probs_train])
                val_mask = np.array([filter_func(p, n_segments) for p in probs_val])
                test_mask = np.array([filter_func(p, n_segments) for p in probs_test])

                print(f"     筛选后：")
                print(f"       训练集保留: {np.sum(train_mask)} / {len(train_mask)} ({np.mean(train_mask)*100:.2f}%)")
                print(f"       验证集保留: {np.sum(val_mask)} / {len(val_mask)} ({np.mean(val_mask)*100:.2f}%)")
                print(f"       测试集保留: {np.sum(test_mask)} / {len(test_mask)} ({np.mean(test_mask)*100:.2f}%)")

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
                print(f"     无筛选，使用全部数据")

            # 检查是否为空
            if len(X_train_filt) == 0 or len(X_val_filt) == 0 or len(X_test_filt) == 0:
                print(f"     ⚠ 筛选后训练集、验证集或测试集为空，跳过该方案")
                continue

            # 构建输出目录：catboost_filtered_pca_<model_name>_<scheme_id>
            output_dir = f'catboost_filtered_pca_{model_name}_{scheme_id}'

            # 训练并评估
            train_and_evaluate(
                X_train_filt.values, y_train_filt.values,
                X_val_filt.values, y_val_filt.values,
                X_test_filt.values, y_test_filt.values,
                feature_names,
                random_state=42,
                scheme_desc=scheme_desc,
                output_dir=output_dir,
                model_tag=model_name
            )

            print(f"\n   方案 {scheme_desc} 完成，结果保存在: {output_dir}")

    print("\n" + "=" * 70)
    print("所有基分类器及方案运行完毕！")
    print("=" * 70)

if __name__ == "__main__":
    main()