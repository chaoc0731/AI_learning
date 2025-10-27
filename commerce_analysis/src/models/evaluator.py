#!/usr/bin/env python3
"""
模型评估器
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve
import joblib

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def comprehensive_evaluation(self, model, X_test, y_test, y_pred, y_pred_proba=None):
        """综合模型评估"""
        logger.info("开始综合模型评估...")

        evaluation_results = {}

        try:
            # 1. 基础指标评估
            evaluation_results['basic_metrics'] = self._evaluate_basic_metrics(y_test, y_pred, y_pred_proba)

            # 2. 分类报告
            evaluation_results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)

            # 3. 混淆矩阵
            evaluation_results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()

            # 4. ROC曲线数据（如果有概率预测）
            if y_pred_proba is not None:
                evaluation_results['roc_analysis'] = self._analyze_roc_curve(y_test, y_pred_proba)
                evaluation_results['pr_analysis'] = self._analyze_pr_curve(y_test, y_pred_proba)

            # 5. 特征重要性分析
            evaluation_results['feature_importance'] = self._analyze_feature_importance(model, X_test)

            # 6. 模型校准分析
            if y_pred_proba is not None:
                evaluation_results['calibration_analysis'] = self._analyze_calibration(y_test, y_pred_proba)

            logger.info("综合模型评估完成")

        except Exception as e:
            logger.error(f"模型评估失败: {e}")

        return evaluation_results

    def _evaluate_basic_metrics(self, y_test, y_pred, y_pred_proba=None):
        """评估基础指标"""
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        # 如果有概率预测，计算AUC
        if y_pred_proba is not None:
            try:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                metrics['auc_score'] = auc(fpr, tpr)
                metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
            except Exception as e:
                logger.warning(f"AUC计算失败: {e}")
                metrics['auc_score'] = 0
                metrics['average_precision'] = 0

        return metrics

    def _analyze_roc_curve(self, y_test, y_pred_proba):
        """分析ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # 找到最佳阈值（Youden指数）
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]

        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': roc_auc,
            'best_threshold': best_threshold,
            'best_tpr': tpr[best_idx],
            'best_fpr': fpr[best_idx]
        }

    def _analyze_pr_curve(self, y_test, y_pred_proba):
        """分析PR曲线"""
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        average_precision = average_precision_score(y_test, y_pred_proba)

        # 找到最佳F1分数的阈值
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # 排除最后一个元素
        best_threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5

        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist(),
            'average_precision': average_precision,
            'best_threshold': best_threshold,
            'best_f1': f1_scores[best_idx] if len(f1_scores) > best_idx else 0
        }

    def _analyze_feature_importance(self, model, X_test, feature_names=None):
        """分析特征重要性"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_

                # 如果没有提供特征名，使用默认的
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(len(importances))]

                # 创建特征重要性DataFrame
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)

                return {
                    'feature_importance': feature_importance_df.to_dict('records'),
                    'top_features': feature_importance_df.head(10).to_dict('records')
                }
            else:
                return {'feature_importance': [], 'top_features': []}

        except Exception as e:
            logger.warning(f"特征重要性分析失败: {e}")
            return {'feature_importance': [], 'top_features': []}

    def _analyze_calibration(self, y_test, y_pred_proba):
        """分析模型校准"""
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_pred_proba, n_bins=10, strategy='uniform'
            )

            return {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        except Exception as e:
            logger.warning(f"模型校准分析失败: {e}")
            return {'fraction_of_positives': [], 'mean_predicted_value': []}

    def create_evaluation_dashboard(self, evaluation_results, save_path=None):
        """创建模型评估仪表盘"""
        logger.info("创建模型评估仪表盘...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Evaluation Dashboard', fontsize=16, fontweight='bold')

        try:
            # 1. 混淆矩阵热力图
            cm = np.array(evaluation_results['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted Label')
            axes[0, 0].set_ylabel('True Label')

            # 2. ROC曲线
            if 'roc_analysis' in evaluation_results:
                roc_data = evaluation_results['roc_analysis']
                axes[0, 1].plot(roc_data['fpr'], roc_data['tpr'],
                                label=f'ROC curve (AUC = {roc_data["auc"]:.3f})', linewidth=2)
                axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
                axes[0, 1].set_xlim([0.0, 1.0])
                axes[0, 1].set_ylim([0.0, 1.05])
                axes[0, 1].set_xlabel('False Positive Rate')
                axes[0, 1].set_ylabel('True Positive Rate')
                axes[0, 1].set_title('ROC Curve')
                axes[0, 1].legend(loc="lower right")
                axes[0, 1].grid(True, alpha=0.3)

            # 3. PR曲线
            if 'pr_analysis' in evaluation_results:
                pr_data = evaluation_results['pr_analysis']
                axes[0, 2].plot(pr_data['recall'], pr_data['precision'],
                                label=f'PR curve (AP = {pr_data["average_precision"]:.3f})', linewidth=2)
                axes[0, 2].set_xlim([0.0, 1.0])
                axes[0, 2].set_ylim([0.0, 1.05])
                axes[0, 2].set_xlabel('Recall')
                axes[0, 2].set_ylabel('Precision')
                axes[0, 2].set_title('Precision-Recall Curve')
                axes[0, 2].legend(loc="upper right")
                axes[0, 2].grid(True, alpha=0.3)

            # 4. 特征重要性
            if 'feature_importance' in evaluation_results:
                feature_data = evaluation_results['feature_importance']
                if feature_data['top_features']:
                    top_features = pd.DataFrame(feature_data['top_features'])
                    axes[1, 0].barh(range(len(top_features)), top_features['importance'])
                    axes[1, 0].set_yticks(range(len(top_features)))
                    axes[1, 0].set_yticklabels(top_features['feature'])
                    axes[1, 0].set_xlabel('Importance')
                    axes[1, 0].set_title('Top 10 Feature Importance')

            # 5. 指标对比
            basic_metrics = evaluation_results['basic_metrics']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metric_values = [
                basic_metrics['accuracy'],
                basic_metrics['precision'],
                basic_metrics['recall'],
                basic_metrics['f1_score']
            ]

            bars = axes[1, 1].bar(metric_names, metric_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Model Performance Metrics')
            axes[1, 1].set_ylim(0, 1)

            # 在柱子上显示数值
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{value:.3f}', ha='center', va='bottom')

            # 6. 模型校准曲线
            if 'calibration_analysis' in evaluation_results:
                cal_data = evaluation_results['calibration_analysis']
                if cal_data['fraction_of_positives'] and cal_data['mean_predicted_value']:
                    axes[1, 2].plot(cal_data['mean_predicted_value'], cal_data['fraction_of_positives'],
                                    's-', label='Model', linewidth=2)
                    axes[1, 2].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=1)
                    axes[1, 2].set_xlabel('Mean Predicted Value')
                    axes[1, 2].set_ylabel('Fraction of Positives')
                    axes[1, 2].set_title('Calibration Plot')
                    axes[1, 2].legend(loc="upper left")
                    axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"模型评估仪表盘已保存至: {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"创建模型评估仪表盘失败: {e}")

    def generate_evaluation_report(self, evaluation_results, output_path):
        """生成详细的评估报告"""
        try:
            report_content = []
            report_content.append("=" * 60)
            report_content.append("模型评估报告")
            report_content.append("=" * 60)
            report_content.append("")

            # 基础指标
            basic_metrics = evaluation_results['basic_metrics']
            report_content.append("基础性能指标:")
            report_content.append(f"  准确率 (Accuracy): {basic_metrics['accuracy']:.4f}")
            report_content.append(f"  精确率 (Precision): {basic_metrics['precision']:.4f}")
            report_content.append(f"  召回率 (Recall): {basic_metrics['recall']:.4f}")
            report_content.append(f"  F1分数: {basic_metrics['f1_score']:.4f}")

            if 'auc_score' in basic_metrics:
                report_content.append(f"  AUC分数: {basic_metrics['auc_score']:.4f}")
                report_content.append(f"  平均精度 (AP): {basic_metrics['average_precision']:.4f}")

            report_content.append("")

            # 分类报告
            class_report = evaluation_results['classification_report']
            report_content.append("详细分类报告:")
            for label, metrics in class_report.items():
                if isinstance(metrics, dict):
                    report_content.append(f"  类别 {label}:")
                    report_content.append(f"    精确率: {metrics['precision']:.4f}")
                    report_content.append(f"    召回率: {metrics['recall']:.4f}")
                    report_content.append(f"    F1分数: {metrics['f1-score']:.4f}")
                    report_content.append(f"    支持数: {metrics['support']}")

            # 特征重要性
            feature_data = evaluation_results['feature_importance']
            if feature_data['top_features']:
                report_content.append("")
                report_content.append("Top 10 重要特征:")
                for i, feature in enumerate(feature_data['top_features'][:10], 1):
                    report_content.append(f"  {i}. {feature['feature']}: {feature['importance']:.4f}")

            # 保存报告
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))

            logger.info(f"评估报告已保存至: {output_path}")

        except Exception as e:
            logger.error(f"生成评估报告失败: {e}")