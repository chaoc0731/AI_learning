#!/usr/bin/env python3
"""
电商用户行为分析系统 - 大数据优化版本
"""

import logging
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    from src.analysis.rfm_analyzer import RFMAnalyzer
    from src.analysis.user_analyzer import UserBehaviorAnalyzer
    from src.analysis.product_analyzer import ProductAnalyzer
    from src.models.predictor import BehaviorPredictor
    from src.models.evaluator import ModelEvaluator
    from src.visualization.dashboard import DashboardCreator
    from commerce_analysis.config.setting import Config

except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖已安装")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.OUTPUT_DIR / 'analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class EcommerceAnalysisPipeline:
    """电商分析流水线 - 大数据优化版本"""

    def __init__(self, use_sample_data: bool = False, sample_ratio: float = 0.05, fast_mode: bool = False):
        self.config = Config
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.rfm_analyzer = RFMAnalyzer()
        self.predictor = BehaviorPredictor()
        self.dashboard_creator = DashboardCreator()
        self.use_sample_data = use_sample_data
        self.sample_ratio = sample_ratio
        self.fast_mode = fast_mode  # 快速模式，跳过耗时分析

    def run_full_analysis(self):
        """运行完整分析流程 - 大数据优化"""
        logger.info("开始电商用户行为分析流程...")

        try:
            # 1. 数据加载
            logger.info("=== 步骤1: 数据加载 ===")
            events_df, item_properties_df, category_tree_df = self.data_loader.load_all_data()

            # 数据质量验证
            if not self.data_loader.validate_data_quality(events_df, item_properties_df, category_tree_df):
                logger.error("数据质量验证失败，终止分析流程")
                return

            # 时间数据诊断
            logger.info("=== 时间数据诊断 ===")
            from src.utils.time_analyzer import TimeDataAnalyzer
            time_diagnostic = TimeDataAnalyzer()
            time_issues = time_diagnostic.diagnose_time_issues(events_df)

            # 根据诊断结果调整预处理策略
            if time_issues.get('missing_timestamp') or time_issues.get('error'):
                logger.error("时间数据存在严重问题，跳过时间相关分析")
                skip_time_analysis = True
            else:
                skip_time_analysis = False

            # 自动检测数据量并调整策略
            data_size = len(events_df)
            logger.info(f"数据集大小: {data_size:,} 条记录")

            # 自动调整采样策略
            if data_size > 1000000 and not self.use_sample_data:
                logger.warning("检测到大数据集，自动启用采样模式")
                self.use_sample_data = True
                self.sample_ratio = 0.05  # 大数据集使用5%样本
            elif data_size > 500000 and not self.use_sample_data:
                logger.info("检测到中等数据集，建议使用采样模式")
                self.use_sample_data = True
                self.sample_ratio = 0.1   # 中等数据集使用10%样本

            # 2. 数据预处理
            logger.info("=== 步骤2: 数据预处理 ===")
            if self.use_sample_data:
                logger.info(f"使用样本数据模式，采样比例: {self.sample_ratio}")
                events_df = self.preprocessor.create_sample_dataset(events_df, self.sample_ratio)
                logger.info(f"采样后数据大小: {len(events_df):,} 条记录")

            processed_events = self.preprocessor.preprocess_events(events_df)

            # 对于大数据集，跳过属性处理以提升性能
            if len(processed_events) > 100000:
                logger.info("数据量较大，跳过商品属性处理")
                processed_properties = self.preprocessor.preprocess_item_properties(item_properties_df)
                merged_data = processed_events  # 不合并属性数据
            else:
                processed_properties = self.preprocessor.preprocess_item_properties(item_properties_df)
                merged_data = self.preprocessor.merge_datasets(
                    processed_events, processed_properties, category_tree_df
                )

            # 保存处理后的数据
            self.preprocessor.save_processed_data(merged_data, 'processed_data')

            # 3. RFM分析（核心分析，必做）
            logger.info("=== 步骤3: RFM分析 ===")
            rfm_base = self.rfm_analyzer.calculate_rfm(processed_events)
            rfm_scored = self.rfm_analyzer.calculate_rfm_scores_robust(rfm_base)
            rfm_segmented = self.rfm_analyzer.segment_users(rfm_scored)

            # 获取分群分析结果
            segment_analysis = self.rfm_analyzer.get_segment_analysis(rfm_segmented)
            print("\n用户分群分析结果:")
            print(segment_analysis)

            # 4. 用户行为分析（根据数据量选择深度）
            logger.info("=== 步骤4: 用户行为分析 ===")
            user_analyzer = UserBehaviorAnalyzer()

            if self.fast_mode or len(processed_events) > 50000:
                logger.info("使用快速用户行为分析模式")
                user_analysis_results = user_analyzer.analyze_behavior_patterns(processed_events)
            else:
                user_analysis_results = user_analyzer.analyze_behavior_patterns(processed_events)

            # 创建用户行为仪表盘（简化版）
            user_analyzer.create_behavior_dashboard(
                processed_events,
                save_path=self.config.OUTPUT_DIR / "figures" / "user_behavior_analysis.png"
            )

            # 5. 商品分析（根据数据量选择深度）
            logger.info("=== 步骤5: 商品分析 ===")
            product_analyzer = ProductAnalyzer()

            if self.fast_mode or len(processed_events) > 50000:
                logger.info("使用快速商品分析模式")
                product_analysis_results = product_analyzer.analyze_products(processed_events, processed_properties)
            else:
                product_analysis_results = product_analyzer.analyze_products(processed_events, processed_properties)

            # 创建商品分析仪表
            product_analyzer.create_product_dashboard(
                processed_events,
                processed_properties,
                save_path=self.config.OUTPUT_DIR / "figures" / "product_analysis.png"
            )

            # 6. 行为预测模型
            logger.info("=== 步骤6: 行为预测模型训练 ===")
            features_df = self.predictor.prepare_features(processed_events, rfm_segmented)

            # 检查是否有足够的交易用户
            transaction_users = features_df['has_transaction'].sum()
            if transaction_users < 50:
                logger.warning(f"交易用户数量较少 ({transaction_users})，跳过模型训练")
                evaluation_results = None
            else:
                X_test, y_test, y_pred = self.predictor.train_model(features_df)

                # 7. 模型评估
                logger.info("=== 步骤7: 模型评估 ===")
                model_evaluator = ModelEvaluator()

                # 获取概率预测（如果模型支持）
                y_pred_proba = None
                if hasattr(self.predictor.model, 'predict_proba'):
                    try:
                        y_pred_proba = self.predictor.model.predict_proba(X_test)[:, 1]
                    except:
                        logger.warning("模型不支持概率预测")

                # 综合模型评估
                evaluation_results = model_evaluator.comprehensive_evaluation(
                    self.predictor.model, X_test, y_test, y_pred, y_pred_proba
                )

                # 创建模型评估仪表盘
                model_evaluator.create_evaluation_dashboard(
                    evaluation_results,
                    save_path=self.config.OUTPUT_DIR / "figures" / "model_evaluation.png"
                )

                # 生成详细评估报告
                model_evaluator.generate_evaluation_report(
                    evaluation_results,
                    output_path=self.config.OUTPUT_DIR / "reports" / "model_evaluation_report.txt"
                )

                # 保存模型
                self.predictor.save_model()

            # 8. 可视化展示
            logger.info("=== 步骤8: 可视化展示 ===")

            # 创建核心仪表盘（必做）
            self.dashboard_creator.create_user_behavior_dashboard(
                processed_events,
                save_path=self.config.OUTPUT_DIR / "figures" / "user_behavior_dashboard.png"
            )

            self.dashboard_creator.create_rfm_dashboard(
                rfm_segmented,
                save_path=self.config.OUTPUT_DIR / "figures" / "rfm_dashboard.png"
            )

            # 交互式仪表盘（大数据集可跳过）
            if len(processed_events) <= 100000:
                logger.info("创建交互式仪表盘...")
                self.dashboard_creator.create_interactive_dashboard(processed_events, rfm_segmented)
            else:
                logger.info("数据量过大，跳过交互式仪表盘")

            logger.info("电商用户行为分析流程完成！")

            # 输出关键洞察
            self._generate_insights(processed_events, rfm_segmented, segment_analysis,
                                   item_properties_df, user_analysis_results, product_analysis_results,
                                   evaluation_results)

        except Exception as e:
            logger.error(f"分析流程执行失败: {e}")
            import traceback
            traceback.print_exc()

    def _generate_insights(self, events_df, rfm_df, segment_analysis, item_properties_df,
                           user_analysis_results, product_analysis_results, evaluation_results):
        """生成业务洞察报告"""
        logger.info("生成业务洞察报告...")

        insights = []

        try:
            # 使用统计计算器
            from src.utils.stat_calculator import StatCalculator
            conversion_stats = StatCalculator.calculate_conversion_rate(events_df)
            user_stats = StatCalculator.calculate_user_activity_stats(events_df)
            time_stats = StatCalculator.calculate_time_patterns(events_df)

        except ImportError:
            logger.warning("统计计算器不可用，使用基础计算")
            conversion_stats = self._basic_conversion_calculation(events_df)
            user_stats = {'total_users': events_df['visitorid'].nunique()}
            time_stats = {'peak_hour': 0, 'weekend_ratio': 0}

        insights.append("📊 总体数据概览:")
        insights.append(f"   - 总用户数: {user_stats.get('total_users', 0):,}")
        insights.append(f"   - 总行为数: {len(events_df):,}")
        insights.append(f"   - 浏览到交易转化率: {conversion_stats.get('view_to_transaction_rate', 0):.2f}%")

        # 模型性能洞察
        if evaluation_results and 'basic_metrics' in evaluation_results:
            basic_metrics = evaluation_results['basic_metrics']
            insights.append("\n🤖 模型性能分析:")
            insights.append(f"   - 模型准确率: {basic_metrics['accuracy']:.3f}")
            insights.append(f"   - 模型精确率: {basic_metrics['precision']:.3f}")
            insights.append(f"   - 模型召回率: {basic_metrics['recall']:.3f}")
            insights.append(f"   - 模型F1分数: {basic_metrics['f1_score']:.3f}")
            if 'auc_score' in basic_metrics:
                insights.append(f"   - 模型AUC分数: {basic_metrics['auc_score']:.3f}")

        # 用户行为洞察
        if user_analysis_results and 'conversion_funnel' in user_analysis_results:
            funnel_data = user_analysis_results['conversion_funnel']['funnel_data']
            insights.append("\n🎯 用户转化漏斗:")
            for _, row in funnel_data.iterrows():
                insights.append(f"   - {row['stage']}: {row['users']:,} 用户 ({row['conversion_rate']:.1f}%)")

        # 商品分析洞察
        if product_analysis_results and 'popular_products' in product_analysis_results:
            popular_products = product_analysis_results['popular_products']
            if 'by_composite_score' in popular_products:
                top_products = list(popular_products['by_composite_score'].items())[:3]
                insights.append("\n🔥 热门商品分析:")
                for i, (product_id, score) in enumerate(top_products, 1):
                    insights.append(f"   - 商品 {product_id}: 综合评分 {score:.3f}")

        # RFM分群洞察
        high_value_users = segment_analysis.loc['高价值用户', '用户数'] if '高价值用户' in segment_analysis.index else 0
        high_value_ratio = segment_analysis.loc['高价值用户', '占比'] if '高价值用户' in segment_analysis.index else 0

        insights.append("\n💎 用户价值分析:")
        insights.append(f"   - 高价值用户: {high_value_users:,} 人 ({high_value_ratio}%)")

        if '潜力用户' in segment_analysis.index:
            insights.append(
                f"   - 潜力用户: {segment_analysis.loc['潜力用户', '用户数']:,} 人 ({segment_analysis.loc['潜力用户', '占比']}%)")

        # 时间模式洞察 - 修复变量引用
        insights.append("\n⏰ 用户行为模式:")

        if 'peak_hour_details' in time_stats:
            peak_info = time_stats['peak_hour_details']
            insights.append(f"   - 用户最活跃时段: {peak_info['hour']:02d}:00 ({peak_info['percentage']:.1f}%的行为)")
            insights.append(f"   - 周末行为占比: {time_stats.get('weekend_ratio', 0):.1f}%")

            # 添加时间分布详情 - 使用 events_df 而不是 processed_events
            hourly_dist = time_stats.get('hourly_distribution', {})
            if hourly_dist:
                # 找出前3个最活跃时段
                top_hours = sorted(hourly_dist.items(), key=lambda x: x[1], reverse=True)[:3]
                insights.append(f"   - 最活跃时段TOP3:")
                for hour, count in top_hours:
                    # 使用 events_df 的总记录数计算百分比
                    percentage = (count / len(events_df)) * 100
                    insights.append(f"     {hour} - {percentage:.1f}%")
        else:
            insights.append(f"   - 用户最活跃时段: {time_stats.get('peak_hour', 0):02d}:00")
            insights.append(f"   - 周末行为占比: {time_stats.get('weekend_ratio', 0):.1f}%")

        # 保存洞察报告
        insights_report = "\n".join(insights)
        report_path = self.config.OUTPUT_DIR / "reports" / "business_insights.txt"

        # 确保目录存在
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("电商用户行为分析 - 业务洞察报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(insights_report)

        print("\n" + "=" * 60)
        print("业务洞察报告")
        print("=" * 60)
        print(insights_report)
        print(f"\n完整报告已保存至: {report_path}")

    def _basic_conversion_calculation(self, events_df):
        """基础转化率计算"""
        try:
            event_counts = events_df['event'].value_counts()
            view_count = event_counts.get('view', 0)
            addtocart_count = event_counts.get('addtocart', 0)
            transaction_count = event_counts.get('transaction', 0)

            view_to_transaction = (transaction_count / view_count * 100) if view_count > 0 else 0
            view_to_cart = (addtocart_count / view_count * 100) if view_count > 0 else 0

            return {
                'view_count': view_count,
                'addtocart_count': addtocart_count,
                'transaction_count': transaction_count,
                'view_to_cart_rate': view_to_cart,
                'view_to_transaction_rate': view_to_transaction
            }
        except:
            return {
                'view_count': 0,
                'addtocart_count': 0,
                'transaction_count': 0,
                'view_to_cart_rate': 0,
                'view_to_transaction_rate': 0
            }

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='电商用户行为分析系统')
    parser.add_argument('--sample', action='store_true', help='使用样本数据进行快速测试')
    parser.add_argument('--sample-ratio', type=float, default=0.1, help='样本数据比例 (默认: 0.1)')
    parser.add_argument('--fast', action='store_true', help='快速模式，跳过耗时分析')

    args = parser.parse_args()

    pipeline = EcommerceAnalysisPipeline(
        use_sample_data=args.sample,
        sample_ratio=args.sample_ratio,
        fast_mode=args.fast
    )
    pipeline.run_full_analysis()

if __name__ == "__main__":
    main()