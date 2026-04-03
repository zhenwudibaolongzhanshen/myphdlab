# -*- coding: utf-8 -*-
"""
对RC-49数据集（PCA特征）进行五种异常检测 + CatBoost回归对比
异常剔除比例：40% (contamination=0.4)，保留60%样本
数据文件：train_features.npy, train_labels.npy, val_features.npy, val_labels.npy, test_features.npy, test_labels.npy
结果保存至：rc49_anomaly_detection_comparison.txt
"""

import numpy as np
import pandas as pd
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

# 设置中文字体（可选）
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


# ==================== 五个异常检测器类（添加 predict 方法）====================
class IsolationForestAnomalyDetector:
    """孤立森林异常检测器"""
    def __init__(self, contamination=0.4, n_estimators=100, random_state=42):
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
        self.median_values_ = None  # 保存训练集各列中位数用于填充

    def preprocess_data(self, df, missing_strategy='fill', median_values=None):
        """
        数据预处理：处理缺失值和无穷值
        如果 median_values 为 None，则计算当前数据的中位数并保存到 self.median_values_
        否则使用提供的 median_values 进行填充（用于验证/测试集）
        """
        print("\n开始数据预处理...")
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = numeric_cols
        print(f"选择的数值特征数量: {len(numeric_cols)}")

        X = df_processed[numeric_cols].copy()

        # 处理无穷值：替换为 NaN
        X = X.replace([np.inf, -np.inf], np.nan)

        # 缺失值处理
        missing_counts = X.isnull().sum().sum()
        if missing_counts > 0:
            print(f"发现缺失值，执行填充策略...")
            if median_values is None:
                # 训练模式：计算中位数并保存
                self.median_values_ = X.median()
                X = X.fillna(self.median_values_)
            else:
                # 预测模式：使用提供的中位数填充
                X = X.fillna(median_values)
            for col in numeric_cols:
                df_processed[col] = X[col]

        print(f"最终数据形状: {X.shape}")
        return X.values, df_processed, X

    def fit_predict(self, X):
        print(f"\n训练孤立森林...")
        self.model.fit(X)
        predictions = self.model.predict(X)          # -1异常, 1正常
        anomaly_scores = self.model.score_samples(X)
        return predictions, anomaly_scores

    def predict(self, X):
        """对新数据进行预测"""
        predictions = self.model.predict(X)
        anomaly_scores = self.model.score_samples(X)
        return predictions, anomaly_scores


class KMeansAnomalyDetector:
    """基于K-means聚类距离的异常检测器"""
    def __init__(self, contamination=0.4, n_clusters=5, use_scaler=True, random_state=42):
        self.contamination = contamination
        self.n_clusters = n_clusters
        self.use_scaler = use_scaler
        self.random_state = random_state
        self.scaler = StandardScaler() if use_scaler else None
        self.model = None
        self.feature_names = None
        self.distance_threshold = None
        self.centers_ = None
        self.median_values_ = None

    def preprocess_data(self, df, missing_strategy='fill', median_values=None):
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = numeric_cols

        X = df_processed[numeric_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        missing_counts = X.isnull().sum().sum()
        if missing_counts > 0:
            if median_values is None:
                self.median_values_ = X.median()
                X = X.fillna(self.median_values_)
            else:
                X = X.fillna(median_values)
            for col in numeric_cols:
                df_processed[col] = X[col]

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
        self.centers_ = self.model.cluster_centers_
        labels = self.model.labels_

        distances = np.zeros(X_scaled.shape[0])
        for i in range(X_scaled.shape[0]):
            distances[i] = np.linalg.norm(X_scaled[i] - self.centers_[labels[i]])

        max_dist = np.max(distances)
        if max_dist > 0:
            anomaly_scores = -distances / max_dist
        else:
            anomaly_scores = -distances

        threshold = np.percentile(distances, (1 - self.contamination) * 100)
        self.distance_threshold = threshold
        predictions = np.where(distances > threshold, -1, 1)
        return predictions, anomaly_scores

    def predict(self, X):
        if self.use_scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        labels = self.model.predict(X_scaled)
        distances = np.zeros(X_scaled.shape[0])
        for i in range(X_scaled.shape[0]):
            distances[i] = np.linalg.norm(X_scaled[i] - self.centers_[labels[i]])

        max_dist = np.max(distances) if len(distances) > 0 else 1
        anomaly_scores = -distances / max_dist if max_dist > 0 else -distances
        predictions = np.where(distances > self.distance_threshold, -1, 1)
        return predictions, anomaly_scores


class IQRAnomalyDetector:
    """基于 IQR 的异常检测器（多变量扩展）"""
    def __init__(self, contamination=0.4, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.feature_names = None
        self.threshold_ = None
        self.medians_ = None
        self.iqrs_ = None
        self.median_values_ = None

    def preprocess_data(self, df, missing_strategy='fill', median_values=None):
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = numeric_cols

        X = df_processed[numeric_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        missing_counts = X.isnull().sum().sum()
        if missing_counts > 0:
            if median_values is None:
                self.median_values_ = X.median()
                X = X.fillna(self.median_values_)
            else:
                X = X.fillna(median_values)
            for col in numeric_cols:
                df_processed[col] = X[col]

        print(f"最终数据形状: {X.shape}")
        return X.values, df_processed, X

    def fit_predict(self, X):
        print(f"\n训练IQR异常检测...")
        self.medians_ = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        self.iqrs_ = q3 - q1
        self.iqrs_ = np.where(self.iqrs_ == 0, 1e-8, self.iqrs_)

        feature_scores = np.abs((X - self.medians_) / self.iqrs_)
        max_feature_scores = np.max(feature_scores, axis=1)
        anomaly_scores = -max_feature_scores

        threshold_raw = np.percentile(max_feature_scores, (1 - self.contamination) * 100)
        self.threshold_ = threshold_raw
        predictions = np.where(max_feature_scores > threshold_raw, -1, 1)
        return predictions, anomaly_scores

    def predict(self, X):
        feature_scores = np.abs((X - self.medians_) / self.iqrs_)
        max_feature_scores = np.max(feature_scores, axis=1)
        anomaly_scores = -max_feature_scores
        predictions = np.where(max_feature_scores > self.threshold_, -1, 1)
        return predictions, anomaly_scores


class HBOSAnomalyDetector:
    """基于直方图的异常检测器 (HBOS)"""
    def __init__(self, contamination=0.4, n_bins=10, alpha=0.1, random_state=None):
        self.contamination = contamination
        self.n_bins = n_bins
        self.alpha = alpha
        self.random_state = random_state
        self.histograms = []  # 存储每个特征的 (bin_edges, bin_densities)
        self.feature_names = None
        self.threshold_ = None
        self.median_values_ = None

    def preprocess_data(self, df, missing_strategy='fill', median_values=None):
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = numeric_cols

        X = df_processed[numeric_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        missing_counts = X.isnull().sum().sum()
        if missing_counts > 0:
            if median_values is None:
                self.median_values_ = X.median()
                X = X.fillna(self.median_values_)
            else:
                X = X.fillna(median_values)
            for col in numeric_cols:
                df_processed[col] = X[col]

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

    def predict(self, X):
        n_samples, n_features = X.shape
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
        predictions = np.where(anomaly_scores <= self.threshold_, -1, 1)
        return predictions, anomaly_scores


class KNNAnomalyDetector:
    """基于KNN距离的异常检测器"""
    def __init__(self, contamination=0.4, n_neighbors=5, random_state=42):
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.neighbors_model = None
        self.threshold_ = None
        self.feature_names = None
        self.median_values_ = None

    def preprocess_data(self, df, missing_strategy='fill', median_values=None):
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = numeric_cols

        X = df_processed[numeric_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        missing_counts = X.isnull().sum().sum()
        if missing_counts > 0:
            if median_values is None:
                self.median_values_ = X.median()
                X = X.fillna(self.median_values_)
            else:
                X = X.fillna(median_values)
            for col in numeric_cols:
                df_processed[col] = X[col]

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

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        distances, _ = self.neighbors_model.kneighbors(X_scaled, n_neighbors=self.n_neighbors+1)
        avg_distances = np.mean(distances[:, 1:], axis=1)
        anomaly_scores = -avg_distances
        predictions = np.where(avg_distances > self.threshold_, -1, 1)
        return predictions, anomaly_scores


# ==================== 主程序 ====================
def main():
    print("=" * 70)
    print("RC-49数据集：五种异常检测 + CatBoost回归对比 (保留60%样本)")
    print("=" * 70)

    # 1. 加载数据
    print("\n加载PCA特征数据...")
    X_train = np.load('train_features.npy')
    y_train = np.load('train_labels.npy')
    X_val = np.load('val_features.npy')
    y_val = np.load('val_labels.npy')
    X_test = np.load('test_features.npy')
    y_test = np.load('test_labels.npy')

    # 转换为DataFrame（便于异常检测预处理）
    feature_names = [f'PC{i+1}' for i in range(X_train.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")

    # 2. 定义所有检测器 (contamination=0.4 表示保留60%正常样本)
    detectors = [
        ("Isolation Forest", IsolationForestAnomalyDetector(contamination=0.4, random_state=42)),
        ("KMeans", KMeansAnomalyDetector(contamination=0.4, n_clusters=5, random_state=42)),
        ("IQR", IQRAnomalyDetector(contamination=0.4, random_state=42)),
        ("HBOS", HBOSAnomalyDetector(contamination=0.4, n_bins=10, alpha=0.1, random_state=42)),
        ("KNN", KNNAnomalyDetector(contamination=0.4, n_neighbors=5, random_state=42)),
    ]

    # 存储结果
    results_list = []

    # CatBoost 参数
    catboost_params = {
        'iterations': 1000,
        'learning_rate': 0.02,
        'depth': 4,
        'loss_function': 'RMSE',
        'eval_metric': 'R2',
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': 100
    }

    # 对每个检测器执行流程
    for name, detector in detectors:
        print(f"\n\n{'='*50}")
        print(f"当前检测器: {name}")
        print('='*50)

        # ---------- 训练集 ----------
        # 预处理（训练模式，会计算并保存中位数）
        X_scaled_train, df_processed_train, X_train_processed = detector.preprocess_data(
            X_train_df, missing_strategy='fill', median_values=None
        )
        # 异常检测（拟合）
        predictions_train, scores_train = detector.fit_predict(X_scaled_train)

        # 正常样本索引
        normal_train_idx = predictions_train == 1
        n_removed_train = np.sum(~normal_train_idx)
        print(f"训练集剔除异常样本数: {n_removed_train} ({n_removed_train/len(predictions_train)*100:.2f}%)")

        # 构建干净训练集
        X_train_clean = X_train[normal_train_idx]
        y_train_clean = y_train[normal_train_idx]

        # ---------- 验证集 ----------
        # 预处理（预测模式，使用训练集保存的中位数）
        X_scaled_val, df_processed_val, X_val_processed = detector.preprocess_data(
            X_val_df, missing_strategy='fill', median_values=detector.median_values_
        )
        predictions_val, scores_val = detector.predict(X_scaled_val)
        normal_val_idx = predictions_val == 1
        n_removed_val = np.sum(~normal_val_idx)
        print(f"验证集剔除异常样本数: {n_removed_val} ({n_removed_val/len(predictions_val)*100:.2f}%)")
        X_val_clean = X_val[normal_val_idx]
        y_val_clean = y_val[normal_val_idx]

        # ---------- 测试集 ----------
        X_scaled_test, df_processed_test, X_test_processed = detector.preprocess_data(
            X_test_df, missing_strategy='fill', median_values=detector.median_values_
        )
        predictions_test, scores_test = detector.predict(X_scaled_test)
        normal_test_idx = predictions_test == 1
        n_removed_test = np.sum(~normal_test_idx)
        print(f"测试集剔除异常样本数: {n_removed_test} ({n_removed_test/len(predictions_test)*100:.2f}%)")
        X_test_clean = X_test[normal_test_idx]
        y_test_clean = y_test[normal_test_idx]

        # ---------- 训练 CatBoost ----------
        model = CatBoostRegressor(**catboost_params)
        model.fit(
            X_train_clean, y_train_clean,
            eval_set=(X_val_clean, y_val_clean),
            use_best_model=True,
            verbose=False
        )

        # 在三个干净集上预测
        train_pred = model.predict(X_train_clean)
        val_pred = model.predict(X_val_clean)
        test_pred = model.predict(X_test_clean)

        # 计算各项指标
        def calc_metrics(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            max_ape = max_absolute_percentage_error(y_true, y_pred)
            return rmse, mae, r2, mape, max_ape

        train_rmse, train_mae, train_r2, train_mape, train_max_ape = calc_metrics(y_train_clean, train_pred)
        val_rmse, val_mae, val_r2, val_mape, val_max_ape = calc_metrics(y_val_clean, val_pred)
        test_rmse, test_mae, test_r2, test_mape, test_max_ape = calc_metrics(y_test_clean, test_pred)

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
        print(f"\n{name} 详细评估结果 (基于剔除异常后的数据集):")
        print(f"{'指标':<15} {'训练集':<15} {'验证集':<15} {'测试集':<15}")
        print("-" * 80)
        print(f"{'RMSE':<15} {train_rmse:<15.4f} {val_rmse:<15.4f} {test_rmse:<15.4f}")
        print(f"{'MAE':<15} {train_mae:<15.4f} {val_mae:<15.4f} {test_mae:<15.4f}")
        print(f"{'R²':<15} {train_r2:<15.4f} {val_r2:<15.4f} {test_r2:<15.4f}")
        print(f"{'MAPE(%)':<15} {train_mape:<15.2f} {val_mape:<15.2f} {test_mape:<15.2f}")
        print(f"{'Max APE(%)':<15} {train_max_ape:<15.2f} {val_max_ape:<15.2f} {test_max_ape:<15.2f}")

    # 汇总结果
    results_df = pd.DataFrame(results_list)
    print("\n\n" + "="*70)
    print("最终对比结果 (RC-49数据集，各数据集均剔除了异常样本，保留60%样本)")
    print("="*70)
    print(results_df.to_string(index=False))

    # 保存到文件
    output_file = "rc49_anomaly_detection_comparison.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("RC-49数据集：五种异常检测 + CatBoost回归对比结果 (各数据集均剔除了异常样本，保留60%样本)\n")
        f.write("="*70 + "\n")
        f.write(results_df.to_string(index=False))
    print(f"\n结果已保存至: {output_file}")


if __name__ == "__main__":
    main()