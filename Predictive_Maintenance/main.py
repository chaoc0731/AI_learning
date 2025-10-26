import sys
import os

sys.path.append('src')

from Predictive_Maintenance.config.setting import Config
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.visualization import DataVisualizer


def main():
    """主函数 - 设备故障预测系统"""
    print("🚀 启动设备故障预测系统...")

    try:
        # 初始化配置和组件
        config = Config()
        data_processor = DataProcessor(config)
        feature_engineer = FeatureEngineer(config)
        model_trainer = ModelTrainer(config)
        evaluator = ModelEvaluator(config)
        visualizer = DataVisualizer(config)

        # 1. 数据处理
        print("\n📊 阶段1: 数据加载与处理")
        df = data_processor.process_data()

        if df is None or df.empty:
            print("❌ 数据加载失败，程序退出")
            return

        # 2. 探索性数据分析 - 使用原始数据
        print("\n📈 阶段2: 探索性数据分析 (原始数据)")
        visualizer.plot_failure_distribution(df)
        visualizer.plot_correlation_heatmap(df)  # 只使用原始特征
        visualizer.plot_feature_distributions(df)

        # 3. 特征工程与模型训练
        print("\n🤖 阶段3: 特征工程与模型训练")
        X, y_binary, y_multi, df_with_features = model_trainer.prepare_data(df, feature_engineer)

        # 检查数据准备是否成功
        if X is None or y_binary is None or y_multi is None:
            print("❌ 特征工程失败，程序退出")
            return

        # 探索性数据分析 - 使用包含工程特征的数据
        print("\n📊 阶段2.5: 探索性数据分析 (包含工程特征)")
        visualizer.plot_enhanced_correlation_heatmap(df_with_features)

        # 获取实际可用的特征名称
        available_features = model_trainer.get_available_features()
        print(f"实际使用的特征: {available_features}")

        # 训练二分类模型
        print("\n--- 训练二分类模型 ---")
        X_test_binary, y_test_binary, y_pred_binary = model_trainer.train_binary_model(X, y_binary)

        if X_test_binary is None:
            print("❌ 二分类模型训练失败，跳过后续步骤")
            return

        # 训练多分类模型
        print("\n--- 训练多分类模型 ---")
        X_test_multi, y_test_multi, y_pred_multi = model_trainer.train_multi_model(X, y_multi)

        if X_test_multi is None:
            print("❌ 多分类模型训练失败，跳过多分类评估")

        # 4. 模型评估
        print("\n📋 阶段4: 模型评估")

        # 二分类模型评估
        binary_metrics = evaluator.evaluate_binary_model(
            model_trainer.binary_model, X_test_binary, y_test_binary, y_pred_binary
        )

        # 多分类模型评估（如果训练成功）
        multi_metrics = {}
        if X_test_multi is not None:
            multi_metrics = evaluator.evaluate_multi_model(
                model_trainer.multi_model, X_test_multi, y_test_multi, y_pred_multi
            )
        else:
            multi_metrics = {'accuracy': 0}

        # 特征重要性分析
        feature_importance = evaluator.feature_importance_analysis(
            model_trainer.binary_model, available_features
        )

        # 5. 风险评分
        print("\n⚠️ 阶段5: 风险评分分析")
        try:
            risk_scores = model_trainer.binary_model.predict_proba(X)[:, 1]
            df_with_risk = visualizer.plot_risk_distribution(df_with_features, risk_scores)
        except Exception as e:
            print(f"风险评分分析时出错: {e}")
            df_with_risk = df_with_features

        # 6. 保存模型和结果
        print("\n💾 阶段6: 保存结果")
        model_trainer.save_models()
        feature_engineer.save_scaler(config.SCALER_PATH)

        # 7. 生成业务报告
        print("\n📄 阶段7: 生成业务报告")
        model_metrics = {
            'binary_accuracy': binary_metrics.get('accuracy', 0),
            'multi_accuracy': multi_metrics.get('accuracy', 0)
        }
        if 'auc_score' in binary_metrics:
            model_metrics['auc_score'] = binary_metrics['auc_score']

        evaluator.generate_business_report(df_with_features, feature_importance, model_metrics, available_features)

        print("\n✅ 设备故障预测系统执行完成！")

    except Exception as e:
        print(f"\n❌ 系统执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()