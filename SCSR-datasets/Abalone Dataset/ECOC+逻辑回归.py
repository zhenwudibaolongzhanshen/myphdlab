# ============================================================
# ECOC-Logistic on Abalone Dataset (with Sex) - 完整版
# 功能：将鲍鱼回归问题转化为6分段分类，使用6×34三值ECOC编码，
#       训练34个逻辑回归二分类器，输出软分类概率及多维度评估指标，
#       并保存模型和所有指标到文件。
# 数据：UCI Abalone，保留Sex列并做独热编码，按70/20/10划分
# 输出目录：ecoc_abalone_with_sex_output_logistic/
# ============================================================

import numpy as np
import pandas as pd
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ==================== 创建输出目录 ====================
output_dir = 'ecoc_abalone_with_sex_output_logistic'
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")

# ==================== 辅助函数：将numpy类型转为Python原生类型 ====================
def convert_to_serializable(obj):
    """递归地将obj中的numpy类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# ==================== 1. 加载并预处理鲍鱼数据集（保留Sex） ====================
print("=" * 60)
print("加载鲍鱼数据集（保留Sex列并进行独热编码）")
print("=" * 60)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
df = pd.read_csv(url, header=None, names=column_names)

# 分离特征和目标
X_df = df.drop('Rings', axis=1)
y = df['Rings'].values

# 对Sex进行独热编码
X_encoded = pd.get_dummies(X_df, columns=['Sex'], prefix=['Sex'])

# 获取新的特征名称列表
feature_names = X_encoded.columns.tolist()
print(f"特征数量: {len(feature_names)}")
print(f"特征名: {feature_names}")

# 转换为numpy数组
X = X_encoded.values

# 划分训练(70%)、验证(20%)、测试(10%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42
)

print(f"训练集: {len(X_train)} 样本")
print(f"验证集: {len(X_val)} 样本")
print(f"测试集: {len(X_test)} 样本")

# ==================== 2. 基于训练集计算6等分边界 ====================
n_segments = 6  # 分段数
quantiles = np.linspace(0, 1, n_segments + 1)
boundaries = np.percentile(y_train, quantiles * 100)
boundaries[0] = np.min(y_train)
boundaries[-1] = np.max(y_train)
print("\n目标值6等分边界（基于训练集）:")
for i, b in enumerate(boundaries):
    print(f"  边界{i}: {b:.4f}")

def assign_segment_labels(y_values, boundaries):
    """根据边界给连续值分配分段标签(0~5)"""
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

y_train_seg = assign_segment_labels(y_train, boundaries)
y_val_seg = assign_segment_labels(y_val, boundaries)
y_test_seg = assign_segment_labels(y_test, boundaries)

print("\n训练集分段分布:")
seg_dist = {}
for seg in range(n_segments):
    count = np.sum(y_train_seg == seg)
    seg_dist[seg] = count
    print(f"  分段{seg}: {count:4d} 样本 ({count/len(y_train_seg)*100:.1f}%)")

# ==================== 3. 预设6×34三值编码矩阵（与之前相同） ====================
code_matrix = np.array([
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0]
], dtype=int)
code_length = code_matrix.shape[1]
print(f"\n编码矩阵: {n_segments}×{code_length} 三值ECOC")

# ==================== 4. 定义ECOC-Logistic训练器类 ====================
class ECOCLogisticTrainer:
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
        """训练所有二分类器"""
        # 标准化
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # 转换为分段标签
        y_train_seg = assign_segment_labels(y_train, self.boundaries)
        y_val_seg = assign_segment_labels(y_val, self.boundaries)

        self.binary_classifiers = []
        self.binary_accuracies = []

        print("\n训练34个二分类器:")
        for col in range(self.code_length):
            print(f"\r  正在训练第 {col+1:2d}/{self.code_length} 个分类器...", end='')
            code_col = self.code_matrix[:, col]
            valid_classes = np.where(code_col != 0)[0]   # 该列涉及的分段

            # 筛选训练集中有效类别的样本
            train_mask = np.isin(y_train_seg, valid_classes)
            if not np.any(train_mask):
                self.binary_classifiers.append(None)
                self.binary_accuracies.append(0.0)
                continue

            X_train_sub = X_train_scaled[train_mask]
            y_train_sub = y_train_seg[train_mask]
            # 二值标签：1 表示code为1，0 表示code为-1
            y_train_bin = (self.code_matrix[y_train_sub, col] == 1).astype(int)

            # 同样处理验证集（用于评估该二分类器性能）
            val_mask = np.isin(y_val_seg, valid_classes)
            if not np.any(val_mask):
                self.binary_classifiers.append(None)
                self.binary_accuracies.append(0.0)
                continue

            X_val_sub = X_val_scaled[val_mask]
            y_val_sub = y_val_seg[val_mask]
            y_val_bin = (self.code_matrix[y_val_sub, col] == 1).astype(int)

            # 检查是否为二分类（是否有两个类）
            if len(np.unique(y_train_bin)) < 2:
                self.binary_classifiers.append(None)
                self.binary_accuracies.append(0.5)
                continue

            # 使用逻辑回归作为基分类器
            lr = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state + col,
                solver='lbfgs'   # 适用于小数据集，也可用 'liblinear'
            )
            lr.fit(X_train_sub, y_train_bin)

            # 验证集准确率
            y_val_pred = lr.predict(X_val_sub)
            acc = accuracy_score(y_val_bin, y_val_pred)

            self.binary_classifiers.append(lr)
            self.binary_accuracies.append(acc)

        print("\n  所有分类器训练完成。")
        valid_accs = [a for a in self.binary_accuracies if a > 0]
        if valid_accs:
            avg_acc = np.mean(valid_accs)
            print(f"  有效二分类器平均准确率: {avg_acc:.4f}")
            self.train_history['binary_avg_accuracy'] = avg_acc
        self.train_history['binary_accuracies'] = self.binary_accuracies

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
        """保存整个训练器对象（含模型、scaler、参数等）"""
        joblib.dump(self, path)
        print(f"\n✅ 完整训练器已保存到: {path}")

# ==================== 5. 定义评估函数（与之前完全相同）====================
def calculate_secondary_accuracy(y_true, y_pred):
    """次准确率：预测正确或相邻都算正确"""
    correct = 0
    for t, p in zip(y_true, y_pred):
        if abs(t - p) <= 1:
            correct += 1
    return correct / len(y_true)

def calculate_adjacent_coherence(probs):
    """邻近一致性：前3概率对应的分段是否相邻"""
    coherent = 0
    n = probs.shape[0]
    for i in range(n):
        top3_idx = np.argsort(probs[i])[-3:][::-1]
        top3_sorted = np.sort(top3_idx)
        if all(top3_sorted[j+1] - top3_sorted[j] <= 1 for j in range(2)):
            coherent += 1
    return coherent / n

def calculate_complete_coverage(y_true, probs):
    """完备覆盖性：前3概率是否包含真实分段及其所有相邻分段"""
    n_segments = probs.shape[1]
    complete = 0
    for i in range(len(y_true)):
        true_seg = y_true[i]
        # 所需覆盖的分段
        if true_seg == 0:
            required = {0, 1}
        elif true_seg == n_segments - 1:
            required = {true_seg-1, true_seg}
        else:
            required = {true_seg-1, true_seg, true_seg+1}
        top3_idx = np.argsort(probs[i])[-3:][::-1]
        if required.issubset(set(top3_idx)):
            complete += 1
    return complete / len(y_true)

def evaluate_and_save(trainer, X, y, name, output_dir):
    """在给定数据集上计算所有指标，并保存到文件"""
    y_seg = assign_segment_labels(y, trainer.boundaries)
    y_pred = trainer.predict(X)
    probs = trainer.predict_proba(X)

    acc = accuracy_score(y_seg, y_pred)
    sec_acc = calculate_secondary_accuracy(y_seg, y_pred)
    adj_coh = calculate_adjacent_coherence(probs)
    comp_cov = calculate_complete_coverage(y_seg, probs)

    # 分段准确率
    seg_accuracies = {}
    for seg in range(trainer.n_segments):
        mask = y_seg == seg
        if np.sum(mask) > 0:
            seg_acc = accuracy_score(y_seg[mask], y_pred[mask])
            seg_accuracies[seg] = seg_acc

    # 混淆矩阵
    cm = confusion_matrix(y_seg, y_pred)
    cm_list = cm.tolist()

    # 打包结果
    results = {
        'dataset': name,
        'accuracy': acc,
        'secondary_accuracy': sec_acc,
        'adjacent_coherence': adj_coh,
        'complete_coverage': comp_cov,
        'segment_accuracies': seg_accuracies,
        'confusion_matrix': cm_list,
        'sample_count': len(y)
    }

    # 转换为可序列化格式
    results_serializable = convert_to_serializable(results)

    # 保存为 JSON
    json_path = os.path.join(output_dir, f'eval_{name}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    print(f"  评估结果已保存: {json_path}")

    # 打印简要信息
    print(f"\n{name} 评估结果:")
    print(f"  主准确率 (完全正确): {acc:.4f}")
    print(f"  次准确率 (±1):      {sec_acc:.4f}")
    print(f"  邻近一致性:         {adj_coh:.4f}")
    print(f"  完备覆盖性:         {comp_cov:.4f}")

    # 打印分段准确率
    print("  分段准确率:")
    for seg in range(trainer.n_segments):
        if seg in seg_accuracies:
            print(f"    分段{seg}: {seg_accuracies[seg]:.4f}")

    return results

# ==================== 6. 训练ECOC-Logistic ====================
trainer = ECOCLogisticTrainer(
    code_matrix, boundaries, n_segments,
    feature_names=feature_names,
    random_state=42
)
trainer.fit(X_train, y_train, X_val, y_val)

# ==================== 7. 评估并保存各数据集结果 ====================
print("\n" + "=" * 60)
print("评估各数据集并保存结果")
print("=" * 60)
results_train = evaluate_and_save(trainer, X_train, y_train, 'train', output_dir)
results_val = evaluate_and_save(trainer, X_val, y_val, 'val', output_dir)
results_test = evaluate_and_save(trainer, X_test, y_test, 'test', output_dir)

# ==================== 8. 保存模型 ====================
model_path = os.path.join(output_dir, 'ecoc_logistic_abalone.pkl')
trainer.save(model_path)

# ==================== 9. 保存汇总指标表 ====================
summary = pd.DataFrame({
    '数据集': ['train', 'val', 'test'],
    '主准确率': [results_train['accuracy'], results_val['accuracy'], results_test['accuracy']],
    '次准确率': [results_train['secondary_accuracy'], results_val['secondary_accuracy'], results_test['secondary_accuracy']],
    '邻近一致性': [results_train['adjacent_coherence'], results_val['adjacent_coherence'], results_test['adjacent_coherence']],
    '完备覆盖性': [results_train['complete_coverage'], results_val['complete_coverage'], results_test['complete_coverage']],
    '样本数': [results_train['sample_count'], results_val['sample_count'], results_test['sample_count']]
})
summary_path = os.path.join(output_dir, 'summary_metrics.csv')
summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
print(f"\n汇总指标表已保存: {summary_path}")

# ==================== 10. 保存训练配置和边界信息 ====================
config = {
    'n_segments': n_segments,
    'code_length': code_length,
    'boundaries': boundaries.tolist(),
    'feature_names': feature_names,
    'train_sample_count': len(X_train),
    'val_sample_count': len(X_val),
    'test_sample_count': len(X_test),
    'segment_distribution': {int(k): int(v) for k, v in seg_dist.items()},
    'binary_classifiers_avg_accuracy': float(trainer.train_history.get('binary_avg_accuracy')) if trainer.train_history.get('binary_avg_accuracy') is not None else None,
    'random_state': 42
}
config_path = os.path.join(output_dir, 'config.json')
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"配置信息已保存: {config_path}")

print("\n" + "=" * 60)
print("所有结果保存完成！")
print(f"输出目录: {output_dir}")
print("=" * 60)