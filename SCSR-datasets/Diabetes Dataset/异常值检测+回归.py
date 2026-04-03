# -*- coding: utf-8 -*-
"""
整合五种异常检测算法与 CatBoost 回归
数据集：糖尿病数据集（sklearn）
异常剔除比例：50% (contamination=0.5)
结果保存至：anomaly_detection_comparison.txt
"""

import numpy as np
import pandas as pd
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

# 设置中文字体（用于可能的绘图，此处未使用但保留）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 评估指标函数 ====================
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


# ==================== 五个异常检测器类（复制自用户代码，稍作调整）====================

class IsolationForestAnomalyDetector:
    """孤立森林异常检测器"""
    def __init__(self, contamination=0.5, n_estimators=100, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = None

    def preprocess_data(self, df, missing_strategy='drop'):
        """数据预处理（只使用数值特征）"""
        print("\n开始数据预处理...")
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        # 排除可能的标识列（糖尿病数据集没有bar_code，但保留安全判断）
        if 'bar_code' in numeric_cols:
            numeric_cols.remove('bar_code')
        print(f"选择的数值特征数量: {len(numeric_cols)}")
        self.feature_names = numeric_cols

        X = df_processed[numeric_cols].copy()

        # 缺失值处理
        missing_counts = X.isnull().sum().sum()
        if missing_counts > 0:
            print(f"发现缺失值，执行{missing_strategy}策略...")
            if missing_strategy == 'drop':
                mask = X.isnull().any(axis=1)
                df_processed = df_processed[~mask].reset_index(drop=True)
                X = X[~mask].reset_index(drop=True)
            elif missing_strategy == 'fill':
                X = X.fillna(X.median())
                for col in numeric_cols:
                    df_processed[col] = X[col]

        # 无穷值处理
        inf_mask = np.isinf(X).any(axis=1)
        if inf_mask.any():
            print(f"发现无穷值，进行删除...")
            df_processed = df_processed[~inf_mask].reset_index(drop=True)
            X = X[~inf_mask].reset_index(drop=True)

        print(f"最终数据形状: {X.shape}")
        return X.values, df_processed, X

    def fit_predict(self, X):
        print(f"\n训练孤立森林...")
        self.model.fit(X)
        predictions = self.model.predict(X)          # -1异常, 1正常
        anomaly_scores = self.model.score_samples(X)
        return predictions, anomaly_scores


class KMeansAnomalyDetector:
    """基于K-means聚类距离的异常检测器"""
    def __init__(self, contamination=0.5, n_clusters=5, use_scaler=True, random_state=42):
        self.contamination = contamination
        self.n_clusters = n_clusters
        self.use_scaler = use_scaler
        self.random_state = random_state
        self.scaler = StandardScaler() if use_scaler else None
        self.model = None
        self.feature_names = None
        self.distance_threshold = None

    def preprocess_data(self, df, missing_strategy='drop'):
        """数据预处理（与上面类似）"""
        print("\n开始数据预处理...")
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if 'bar_code' in numeric_cols:
            numeric_cols.remove('bar_code')
        self.feature_names = numeric_cols

        X = df_processed[numeric_cols].copy()

        missing_counts = X.isnull().sum().sum()
        if missing_counts > 0:
            if missing_strategy == 'drop':
                mask = X.isnull().any(axis=1)
                df_processed = df_processed[~mask].reset_index(drop=True)
                X = X[~mask].reset_index(drop=True)
            elif missing_strategy == 'fill':
                X = X.fillna(X.median())
                for col in numeric_cols:
                    df_processed[col] = X[col]

        inf_mask = np.isinf(X).any(axis=1)
        if inf_mask.any():
            df_processed = df_processed[~inf_mask].reset_index(drop=True)
            X = X[~inf_mask].reset_index(drop=True)

        print(f"最终数据形状: {X.shape}")
        return X.values, df_processed, X

    def fit_predict(self, X):
        print(f"\n训练K-means异常检测...")
        if self.use_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.model.fit(X_scaled)
        centers = self.model.cluster_centers_
        labels = self.model.labels_

        distances = np.zeros(X_scaled.shape[0])
        for i in range(X_scaled.shape[0]):
            distances[i] = np.linalg.norm(X_scaled[i] - centers[labels[i]])

        max_dist = np.max(distances)
        if max_dist > 0:
            anomaly_scores = -distances / max_dist
        else:
            anomaly_scores = -distances

        threshold = np.percentile(distances, (1 - self.contamination) * 100)
        self.distance_threshold = threshold
        predictions = np.where(distances > threshold, -1, 1)
        return predictions, anomaly_scores


class IQRAnomalyDetector:
    """基于 IQR 的异常检测器（多变量扩展）"""
    def __init__(self, contamination=0.5, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.feature_names = None
        self.threshold_ = None

    def preprocess_data(self, df, missing_strategy='drop'):
        print("\n开始数据预处理...")
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if 'bar_code' in numeric_cols:
            numeric_cols.remove('bar_code')
        self.feature_names = numeric_cols

        X = df_processed[numeric_cols].copy()

        missing_counts = X.isnull().sum().sum()
        if missing_counts > 0:
            if missing_strategy == 'drop':
                mask = X.isnull().any(axis=1)
                df_processed = df_processed[~mask].reset_index(drop=True)
                X = X[~mask].reset_index(drop=True)
            elif missing_strategy == 'fill':
                X = X.fillna(X.median())
                for col in numeric_cols:
                    df_processed[col] = X[col]

        inf_mask = np.isinf(X).any(axis=1)
        if inf_mask.any():
            df_processed = df_processed[~inf_mask].reset_index(drop=True)
            X = X[~inf_mask].reset_index(drop=True)

        print(f"最终数据形状: {X.shape}")
        return X.values, df_processed, X

    def fit_predict(self, X):
        print(f"\n训练IQR异常检测...")
        medians = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqrs = q3 - q1
        iqrs = np.where(iqrs == 0, 1e-8, iqrs)

        feature_scores = np.abs((X - medians) / iqrs)
        max_feature_scores = np.max(feature_scores, axis=1)
        anomaly_scores = -max_feature_scores

        threshold_raw = np.percentile(max_feature_scores, (1 - self.contamination) * 100)
        self.threshold_ = threshold_raw
        predictions = np.where(max_feature_scores > threshold_raw, -1, 1)
        return predictions, anomaly_scores


class HBOSAnomalyDetector:
    """基于直方图的异常检测器 (HBOS)"""
    def __init__(self, contamination=0.5, n_bins=10, alpha=0.1, random_state=None):
        self.contamination = contamination
        self.n_bins = n_bins
        self.alpha = alpha
        self.random_state = random_state
        self.histograms = []
        self.feature_names = None
        self.threshold_ = None

    def preprocess_data(self, df, missing_strategy='drop'):
        print("\n开始数据预处理...")
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if 'bar_code' in numeric_cols:
            numeric_cols.remove('bar_code')
        self.feature_names = numeric_cols

        X = df_processed[numeric_cols].copy()

        missing_counts = X.isnull().sum().sum()
        if missing_counts > 0:
            if missing_strategy == 'drop':
                mask = X.isnull().any(axis=1)
                df_processed = df_processed[~mask].reset_index(drop=True)
                X = X[~mask].reset_index(drop=True)
            elif missing_strategy == 'fill':
                X = X.fillna(X.median())
                for col in numeric_cols:
                    df_processed[col] = X[col]

        inf_mask = np.isinf(X).any(axis=1)
        if inf_mask.any():
            df_processed = df_processed[~inf_mask].reset_index(drop=True)
            X = X[~inf_mask].reset_index(drop=True)

        print(f"最终数据形状: {X.shape}")
        return X.values, df_processed, X

    def fit_predict(self, X):
        print(f"\n训练HBOS异常检测...")
        n_samples, n_features = X.shape
        self.histograms = []

        for i in range(n_features):
            feature_data = X[:, i]
            hist, bin_edges = np.histogram(feature_data, bins=self.n_bins, density=False)
            bin_counts = hist + self.alpha
            bin_densities = bin_counts / (np.sum(bin_counts) * np.diff(bin_edges))
            self.histograms.append((bin_edges, bin_densities))

        anomaly_scores = np.zeros(n_samples)
        for i in range(n_samples):
            score = 0.0
            for j in range(n_features):
                value = X[i, j]
                bin_edges, bin_densities = self.histograms[j]
                bin_idx = np.digitize(value, bin_edges, right=False) - 1
                bin_idx = max(0, min(bin_idx, len(bin_densities)-1))
                density = bin_densities[bin_idx]
                score += -np.log(density + 1e-12)
            anomaly_scores[i] = score / n_features

        anomaly_scores = -anomaly_scores
        threshold = np.percentile(anomaly_scores, self.contamination * 100)
        self.threshold_ = threshold
        predictions = np.where(anomaly_scores <= threshold, -1, 1)
        return predictions, anomaly_scores


class KNNAnomalyDetector:
    """基于KNN距离的异常检测器"""
    def __init__(self, contamination=0.5, n_neighbors=5, random_state=42):
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.neighbors_model = None
        self.threshold_ = None
        self.feature_names = None

    def preprocess_data(self, df, missing_strategy='drop'):
        print("\n开始数据预处理...")
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if 'bar_code' in numeric_cols:
            numeric_cols.remove('bar_code')
        self.feature_names = numeric_cols

        X = df_processed[numeric_cols].copy()

        missing_counts = X.isnull().sum().sum()
        if missing_counts > 0:
            if missing_strategy == 'drop':
                mask = X.isnull().any(axis=1)
                df_processed = df_processed[~mask].reset_index(drop=True)
                X = X[~mask].reset_index(drop=True)
            elif missing_strategy == 'fill':
                X = X.fillna(X.median())
                for col in numeric_cols:
                    df_processed[col] = X[col]

        inf_mask = np.isinf(X).any(axis=1)
        if inf_mask.any():
            df_processed = df_processed[~inf_mask].reset_index(drop=True)
            X = X[~inf_mask].reset_index(drop=True)

        print(f"最终数据形状: {X.shape}")
        return X.values, df_processed, X

    def fit_predict(self, X):
        print(f"\n训练KNN异常检测...")
        X_scaled = self.scaler.fit_transform(X)
        self.neighbors_model = NearestNeighbors(n_neighbors=self.n_neighbors+1, n_jobs=-1)
        self.neighbors_model.fit(X_scaled)
        distances, _ = self.neighbors_model.kneighbors(X_scaled, n_neighbors=self.n_neighbors+1)
        avg_distances = np.mean(distances[:, 1:], axis=1)
        anomaly_scores = -avg_distances
        threshold = np.percentile(avg_distances, (1 - self.contamination) * 100)
        self.threshold_ = threshold
        predictions = np.where(avg_distances > threshold, -1, 1)
        return predictions, anomaly_scores


# ==================== 主程序 ====================
def main():
    print("=" * 70)
    print("异常检测 + CatBoost 回归对比实验")
    print("=" * 70)

    # 1. 加载糖尿病数据集
    data = load_diabetes()
    X, y = data.data, data.target
    feature_names = data.feature_names
    print(f"\n数据集样本数: {X.shape[0]}, 特征数: {X.shape[1]}")

    # 2. 划分训练 (70%)、验证 (20%)、测试 (10%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42
    )
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")

    # 将训练特征转换为 DataFrame（便于异常检测预处理）
    X_train_df = pd.DataFrame(X_train, columns=feature_names)

    # 3. 定义所有检测器
    detectors = [
        ("Isolation Forest", IsolationForestAnomalyDetector(contamination=0.5, random_state=42)),
        ("KMeans", KMeansAnomalyDetector(contamination=0.5, n_clusters=5, random_state=42)),
        ("IQR", IQRAnomalyDetector(contamination=0.5, random_state=42)),
        ("HBOS", HBOSAnomalyDetector(contamination=0.5, n_bins=10, alpha=0.1, random_state=42)),
        ("KNN", KNNAnomalyDetector(contamination=0.5, n_neighbors=5, random_state=42)),
    ]

    # 存储结果
    results_list = []

    # 基础 CatBoost 参数（与原始一致）
    catboost_params = {
        'iterations': 8000,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'RMSE',
        'eval_metric': 'R2',
        'random_seed': 42,
        'verbose': 0,          # 不输出中间日志，保持简洁
        'early_stopping_rounds': 100
    }

    # 对每个检测器执行流程
    for name, detector in detectors:
        print(f"\n\n{'='*50}")
        print(f"当前检测器: {name}")
        print('='*50)

        # 4. 在训练集上预处理（无缺失值，基本不变）
        X_scaled, df_processed, _ = detector.preprocess_data(X_train_df, missing_strategy='drop')

        # 5. 异常检测
        predictions, scores = detector.fit_predict(X_scaled)

        # 正常样本索引
        normal_idx = predictions == 1
        n_removed = np.sum(~normal_idx)
        print(f"剔除异常样本数: {n_removed} ({n_removed/len(predictions)*100:.2f}%)")

        # 6. 获取干净训练集
        X_train_clean = X_train[normal_idx]
        y_train_clean = y_train[normal_idx]

        # 7. 训练 CatBoost
        model = CatBoostRegressor(**catboost_params)
        model.fit(
            X_train_clean, y_train_clean,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False
        )

        # 8. 在三个集上预测
        train_pred = model.predict(X_train_clean)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        # 9. 计算各项指标
        def calc_metrics(y_true, y_pred, name_prefix=""):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            max_ape = max_absolute_percentage_error(y_true, y_pred)
            return rmse, mae, r2, mape, max_ape

        train_rmse, train_mae, train_r2, train_mape, train_max_ape = calc_metrics(y_train_clean, train_pred)
        val_rmse, val_mae, val_r2, val_mape, val_max_ape = calc_metrics(y_val, val_pred)
        test_rmse, test_mae, test_r2, test_mape, test_max_ape = calc_metrics(y_test, test_pred)

        # 记录结果
        results_list.append({
            'Algorithm': name,
            'Train_Size': len(X_train_clean),
            'Val_RMSE': val_rmse,
            'Test_RMSE': test_rmse,
            'Val_MAE': val_mae,
            'Test_MAE': test_mae,
            'Val_R2': val_r2,
            'Test_R2': test_r2,
            'Val_MAPE(%)': val_mape,
            'Test_MAPE(%)': test_mape,
            'Val_MaxAPE(%)': val_max_ape,
            'Test_MaxAPE(%)': test_max_ape,
            'Iterations': model.tree_count_
        })

        # 打印当前结果
        print(f"\n{name} 结果:")
        print(f"  训练集大小: {len(X_train_clean)} (原始 {len(X_train)})")
        print(f"  Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"  Val R²  : {val_r2:.4f}, Test R²  : {test_r2:.4f}")
        print(f"  Val MAPE: {val_mape:.2f}%, Test MAPE: {test_mape:.2f}%")

    # 10. 汇总结果
    results_df = pd.DataFrame(results_list)
    print("\n\n" + "="*70)
    print("最终对比结果")
    print("="*70)
    print(results_df.to_string(index=False))

    # 保存到文件
    output_file = "anomaly_detection_comparison.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("异常检测 + CatBoost 回归对比结果\n")
        f.write("="*70 + "\n")
        f.write(results_df.to_string(index=False))
    print(f"\n结果已保存至: {output_file}")


if __name__ == "__main__":
    main()