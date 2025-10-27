#!/usr/bin/env python3
"""
诊断product_analyzer和user_analyzer对时间分析的影响
"""

import sys
from pathlib import Path
import pandas as pd
import logging

sys.path.append(str(Path(__file__).parent / 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_analyzer_impact():
    """诊断分析器对时间分析的影响"""
    print("=" * 60)
    print("分析器影响诊断")
    print("=" * 60)

    try:
        from commerce_analysis.src.data.loader import DataLoader
        from commerce_analysis.src.data.preprocessor import DataPreprocessor

        # 1. 加载和预处理数据
        loader = DataLoader()
        events_df, item_properties_df, _ = loader.load_all_data()

        # 采样
        preprocessor = DataPreprocessor()
        events_df = preprocessor.create_sample_dataset(events_df, 0.01)
        processed_events = preprocessor.preprocess_events(events_df)

        print("1. 预处理后的时间特征:")
        print(f"   hour列存在: {'hour' in processed_events.columns}")
        print(f"   weekend列存在: {'weekend' in processed_events.columns}")
        if 'hour' in processed_events.columns:
            print(f"   hour唯一值: {sorted(processed_events['hour'].unique())}")
        if 'weekend' in processed_events.columns:
            print(f"   weekend唯一值: {sorted(processed_events['weekend'].unique())}")

        # 2. 测试直接时间分析
        print("\n2. 直接时间分析结果:")
        peak_hour = processed_events['hour'].mode().iloc[0] if not processed_events['hour'].mode().empty else 0
        weekend_ratio = processed_events['weekend'].mean() * 100
        print(f"   最活跃时段: {peak_hour:02d}:00")
        print(f"   周末占比: {weekend_ratio:.2f}%")

        # 3. 测试user_analyzer的影响
        print("\n3. 测试user_analyzer...")
        try:
            from commerce_analysis.src.analysis.user_analyzer import UserBehaviorAnalyzer
            user_analyzer = UserBehaviorAnalyzer()

            # 检查analyzer是否修改数据
            events_before = processed_events.copy()
            user_analysis_results = user_analyzer.analyze_behavior_patterns(processed_events)
            events_after = processed_events.copy()

            # 比较数据是否被修改
            hour_changed = not events_before['hour'].equals(events_after['hour'])
            weekend_changed = not events_before['weekend'].equals(events_after['weekend'])

            print(f"   user_analyzer是否修改hour列: {hour_changed}")
            print(f"   user_analyzer是否修改weekend列: {weekend_changed}")

            # 检查user_analyzer返回的时间分析
            if 'time_patterns' in user_analysis_results:
                user_time = user_analysis_results['time_patterns']
                print(
                    f"   user_analyzer时间分析: peak_hour={user_time.get('peak_hour', 'N/A')}, weekend_ratio={user_time.get('weekend_ratio', 'N/A')}")

        except Exception as e:
            print(f"   user_analyzer测试失败: {e}")

        # 4. 测试product_analyzer的影响
        print("\n4. 测试product_analyzer...")
        try:
            from commerce_analysis.src.analysis.product_analyzer import ProductAnalyzer
            product_analyzer = ProductAnalyzer()

            events_before = processed_events.copy()
            product_analysis_results = product_analyzer.analyze_products(processed_events, pd.DataFrame())
            events_after = processed_events.copy()

            hour_changed = not events_before['hour'].equals(events_after['hour'])
            weekend_changed = not events_before['weekend'].equals(events_after['weekend'])

            print(f"   product_analyzer是否修改hour列: {hour_changed}")
            print(f"   product_analyzer是否修改weekend列: {weekend_changed}")

        except Exception as e:
            print(f"   product_analyzer测试失败: {e}")

    except Exception as e:
        print(f"诊断失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    diagnose_analyzer_impact()