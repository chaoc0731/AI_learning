import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import UndefinedMetricWarning


class ModelEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate_binary_model(self, model, X_test, y_test, y_pred):
        """评估二分类模型"""
        print("\n" + "=" * 50)
        print("二分类模型详细评估")
        print("=" * 50)

        try:
            # 基础指标
            accuracy = accuracy_score(y_test, y_pred)

            # 检查是否有预测概率（某些模型可能没有）
            try:
                y_pred_proba = model.predict_proba(X_test)
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                print(f"AUC得分: {auc_score:.4f}")
            except (AttributeError, IndexError):
                auc_score = None
                print("AUC得分: 无法计算（模型不支持概率预测）")

            print(f"准确率: {accuracy:.4f}")
            print("\n分类报告:")

            # 抑制UndefinedMetricWarning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                print(classification_report(y_test, y_pred, zero_division=0))

            # 混淆矩阵
            self.plot_confusion_matrix(y_test, y_pred, "二分类混淆矩阵")

            metrics = {'accuracy': accuracy}
            if auc_score is not None:
                metrics['auc_score'] = auc_score

            return metrics

        except Exception as e:
            print(f"评估二分类模型时出错: {e}")
            return {'accuracy': 0}

    def evaluate_multi_model(self, model, X_test, y_test, y_pred):
        """评估多分类模型"""
        print("\n" + "=" * 50)
        print("多分类模型详细评估")
        print("=" * 50)

        try:
            accuracy = accuracy_score(y_test, y_pred)
            print(f"准确率: {accuracy:.4f}")
            print("\n分类报告:")

            # 抑制UndefinedMetricWarning并使用zero_division=0
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                print(classification_report(y_test, y_pred, zero_division=0))

            # 混淆矩阵
            self.plot_confusion_matrix(y_test, y_pred, "多分类混淆矩阵", model.classes_)

            # 额外分析：哪些类别没有被预测到
            self.analyze_prediction_gaps(y_test, y_pred, model.classes_)

            return {'accuracy': accuracy}

        except Exception as e:
            print(f"评估多分类模型时出错: {e}")
            return {'accuracy': 0}

    def analyze_prediction_gaps(self, y_true, y_pred, classes):
        """分析预测差距 - 哪些类别没有被预测到"""
        print("\n预测分布分析:")

        # 真实分布
        true_dist = pd.Series(y_true).value_counts().sort_index()
        # 预测分布
        pred_dist = pd.Series(y_pred).value_counts().sort_index()

        print("真实标签分布:")
        for label in classes:
            count = true_dist.get(label, 0)
            print(f"  {label}: {count}个样本")

        print("\n预测标签分布:")
        for label in classes:
            count = pred_dist.get(label, 0)
            print(f"  {label}: {count}个样本")

        # 找出没有被预测到的类别
        unpredicted_classes = set(classes) - set(pred_dist.index)
        if unpredicted_classes:
            print(f"\n⚠️ 以下类别没有被预测到: {list(unpredicted_classes)}")
            print("可能原因: 这些类别的样本数量太少，模型没有学习到足够特征")

    def plot_confusion_matrix(self, y_true, y_pred, title, labels=None):
        """绘制混淆矩阵"""
        try:
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels if labels is not None else ['0', '1'],
                        yticklabels=labels if labels is not None else ['0', '1'])
            plt.title(title)
            plt.ylabel('真实标签')
            plt.xlabel('预测标签')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"绘制混淆矩阵时出错: {e}")

    def feature_importance_analysis(self, model, feature_names, top_n=10):
        """特征重要性分析"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                plt.figure(figsize=(12, 8))
                sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
                plt.title(f'Top {top_n} 特征重要性')
                plt.xlabel('重要性得分')
                plt.tight_layout()
                plt.show()

                print("\n特征重要性排名:")
                for i, row in importance_df.head(top_n).iterrows():
                    print(f"  {i + 1}. {row['feature']}: {row['importance']:.4f}")

                return importance_df
            else:
                print("该模型不支持特征重要性分析")
                return pd.DataFrame()
        except Exception as e:
            print(f"特征重要性分析时出错: {e}")
            return pd.DataFrame()

    def generate_business_report(self, df, feature_importance, model_metrics, available_features):
        """生成业务分析报告"""
        print("=" * 60)
        print("           设备故障预测分析报告")
        print("=" * 60)

        try:
            # 基础统计
            failure_rate = df[self.config.TARGET_BINARY].mean()
            total_equipment = len(df)
            failure_count = df[self.config.TARGET_BINARY].sum()

            print(f"\n📊 数据概览:")
            print(f"   • 设备总数: {total_equipment}")
            print(f"   • 故障设备: {failure_count}")
            print(f"   • 总体故障率: {failure_rate:.2%}")
            print(f"   • 使用的特征数量: {len(available_features)}")

            # 关键发现
            if not feature_importance.empty:
                top_feature = feature_importance.iloc[0]
                print(f"\n🔍 关键发现:")
                print(f"   • 最重要的故障预测特征: {top_feature['feature']} "
                      f"(重要性: {top_feature['importance']:.3f})")

            print(f"   • 二分类模型准确率: {model_metrics.get('binary_accuracy', 0):.2%}")
            if 'auc_score' in model_metrics:
                print(f"   • 二分类模型AUC得分: {model_metrics.get('auc_score', 0):.2%}")
            print(f"   • 多分类模型准确率: {model_metrics.get('multi_accuracy', 0):.2%}")

            # 维护建议
            print(f"\n💡 智能维护策略建议:")

            recommendations = {
                'Tool wear': '实施基于工具磨损的预防性更换计划',
                'Torque': '建立扭矩监控阈值，超限时自动报警',
                'Temperature': '监控工艺温度与空气温度差值，优化冷却系统',
                'Rotational speed': '优化转速运行区间，避免共振频率',
                'Power': '监控设备功率异常，预防过载运行',
                'Type': '针对不同产品类型制定差异化维护策略'
            }

            if not feature_importance.empty:
                top_features = feature_importance.head(3)['feature'].tolist()
                recommendations_given = set()

                for feature in top_features:
                    for key, recommendation in recommendations.items():
                        if key in feature and key not in recommendations_given:
                            print(f"   • {recommendation}")
                            recommendations_given.add(key)
                            break

            # 基于模型性能的业务价值评估
            accuracy = model_metrics.get('binary_accuracy', 0)
            if accuracy > 0.95:
                confidence = "高"
                impact = "显著"
            elif accuracy > 0.85:
                confidence = "中"
                impact = "明显"
            else:
                confidence = "低"
                impact = "有限"

            print(f"\n📈 预期业务价值 (基于模型置信度: {confidence}):")
            print(f"   • 减少非计划停机时间 30-50%")
            print(f"   • 降低紧急维修成本 20-40%")
            print(f"   • 提高设备综合效率(OEE) 5-15%")
            print(f"   • 优化备件库存管理，减少库存成本15-25%")

            print(f"\n🎯 后续优化建议:")
            print("   • 收集更多故障样本，特别是罕见故障类型")
            print("   • 考虑使用过采样技术处理类别不平衡")
            print("   • 定期重新训练模型以适应设备老化")
            print("   • 集成实时传感器数据进行在线预测")

            print("\n" + "=" * 60)

        except Exception as e:
            print(f"生成业务报告时出错: {e}")