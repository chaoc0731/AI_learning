#!/usr/bin/env python3
"""
时间数据分析工具
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TimeDataAnalyzer:
    """时间数据分析器"""

    @staticmethod
    def diagnose_time_issues(events_df):
        """诊断时间数据问题"""
        print("=" * 60)
        print("时间数据诊断报告")
        print("=" * 60)

        diagnostics = {}

        try:
            # 1. 检查timestamp列是否存在
            if 'timestamp' not in events_df.columns:
                print("❌ 错误: 数据中缺少timestamp列")
                diagnostics['missing_timestamp'] = True
                return diagnostics

            diagnostics['missing_timestamp'] = False

            # 2. 检查数据类型
            timestamp_dtype = events_df['timestamp'].dtype
            print(f"timestamp数据类型: {timestamp_dtype}")
            diagnostics['timestamp_dtype'] = str(timestamp_dtype)

            # 3. 显示时间戳示例
            print("\n时间戳前5个示例:")
            for i, ts in enumerate(events_df['timestamp'].head()):
                print(f"  {i + 1}. {ts} (类型: {type(ts)})")

            # 4. 检查时间范围
            if pd.api.types.is_datetime64_any_dtype(events_df['timestamp']):
                time_range = {
                    'min': events_df['timestamp'].min(),
                    'max': events_df['timestamp'].max(),
                    'range_days': (events_df['timestamp'].max() - events_df['timestamp'].min()).days
                }
                print(f"\n时间范围: {time_range['min']} 到 {time_range['max']}")
                print(f"时间跨度: {time_range['range_days']} 天")
                diagnostics['time_range'] = time_range
            else:
                print("❌ timestamp列不是datetime类型")
                diagnostics['is_datetime'] = False

            # 5. 检查空值
            null_count = events_df['timestamp'].isna().sum()
            total_count = len(events_df)
            null_percentage = (null_count / total_count) * 100
            print(f"\n空值检查: {null_count}/{total_count} ({null_percentage:.2f}%)")
            diagnostics['null_info'] = {
                'null_count': null_count,
                'total_count': total_count,
                'null_percentage': null_percentage
            }

            # 6. 检查小时分布
            if pd.api.types.is_datetime64_any_dtype(events_df['timestamp']):
                events_df = events_df.copy()
                events_df['hour'] = events_df['timestamp'].dt.hour
                hour_dist = events_df['hour'].value_counts().sort_index()

                print(f"\n小时分布:")
                for hour, count in hour_dist.items():
                    percentage = (count / total_count) * 100
                    print(f"  {hour:02d}:00 - {count:>8,} 次 ({percentage:>5.1f}%)")

                diagnostics['hour_distribution'] = hour_dist.to_dict()

                # 检查0点是否异常
                hour_0_count = hour_dist.get(0, 0)
                hour_0_percentage = (hour_0_count / total_count) * 100
                if hour_0_percentage > 50:
                    print(f"⚠️  警告: 0点行为占比过高 ({hour_0_percentage:.1f}%)，可能存在数据问题")
                    diagnostics['hour_0_issue'] = True
                else:
                    diagnostics['hour_0_issue'] = False

            # 7. 检查星期分布
            if pd.api.types.is_datetime64_any_dtype(events_df['timestamp']):
                events_df['day_of_week'] = events_df['timestamp'].dt.dayofweek
                weekday_dist = events_df['day_of_week'].value_counts().sort_index()

                weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
                print(f"\n星期分布:")
                for day, count in weekday_dist.items():
                    day_name = weekday_names[day] if day < len(weekday_names) else f'周{day + 1}'
                    percentage = (count / total_count) * 100
                    print(f"  {day_name} - {count:>8,} 次 ({percentage:>5.1f}%)")

                diagnostics['weekday_distribution'] = weekday_dist.to_dict()

                # 检查周末占比
                weekend_mask = events_df['day_of_week'].isin([5, 6])
                weekend_count = weekend_mask.sum()
                weekend_percentage = (weekend_count / total_count) * 100
                print(f"\n周末行为占比: {weekend_count:,}/{total_count:,} = {weekend_percentage:.2f}%")

                if weekend_percentage < 10:
                    print("⚠️  警告: 周末行为占比过低，可能存在数据问题")
                    diagnostics['low_weekend_issue'] = True
                else:
                    diagnostics['low_weekend_issue'] = False

            return diagnostics

        except Exception as e:
            print(f"❌ 时间数据诊断失败: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
