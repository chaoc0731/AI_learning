#!/usr/bin/env python3
"""
时间戳验证工具
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TimestampValidator:
    """时间戳验证器"""

    @staticmethod
    def validate_timestamps(events_df):
        """验证时间戳数据"""
        print("=" * 60)
        print("时间戳验证报告")
        print("=" * 60)

        if 'timestamp' not in events_df.columns:
            print("❌ 错误: 数据中缺少timestamp列")
            return None

        # 原始时间戳分析
        original_timestamps = events_df['timestamp']
        print(f"原始时间戳类型: {original_timestamps.dtype}")
        print(f"时间戳范围: {original_timestamps.min()} 到 {original_timestamps.max()}")

        # 检查时间戳长度（判断是秒还是毫秒）
        sample_ts = original_timestamps.iloc[0]
        ts_length = len(str(sample_ts))
        print(f"时间戳长度: {ts_length} 位")

        if ts_length == 13:
            print("✅ 时间戳为13位，很可能是毫秒时间戳")
            unit = 'ms'
        elif ts_length == 10:
            print("✅ 时间戳为10位，很可能是秒时间戳")
            unit = 's'
        else:
            print(f"⚠️  时间戳为{ts_length}位，长度异常")
            unit = 'ms'  # 默认按毫秒处理

        # 转换时间戳
        print(f"\n尝试按{unit}单位转换时间戳...")
        try:
            if unit == 'ms':
                converted_timestamps = pd.to_datetime(original_timestamps, unit='ms')
            else:
                converted_timestamps = pd.to_datetime(original_timestamps, unit='s')

            # 检查转换结果
            valid_count = converted_timestamps.notna().sum()
            invalid_count = converted_timestamps.isna().sum()
            total_count = len(converted_timestamps)

            print(f"转换结果: {valid_count}/{total_count} 成功, {invalid_count} 失败")

            if valid_count > 0:
                # 分析转换后的时间
                min_time = converted_timestamps.min()
                max_time = converted_timestamps.max()
                time_span = max_time - min_time

                print(f"\n转换后时间范围:")
                print(f"  开始时间: {min_time}")
                print(f"  结束时间: {max_time}")
                print(f"  时间跨度: {time_span.days} 天")
                print(f"  数据年份: {min_time.year} - {max_time.year}")

                # 检查时间合理性
                current_year = datetime.now().year
                data_years = converted_timestamps.dt.year.unique()
                print(f"  包含年份: {sorted(data_years)}")

                if min_time.year < 2000:
                    print("⚠️  警告: 时间数据早于2000年，可能不合理")
                if max_time.year > current_year + 1:
                    print("⚠️  警告: 时间数据晚于当前年份，可能不合理")

                # 分析时间分布
                print(f"\n时间分布分析:")
                hour_dist = converted_timestamps.dt.hour.value_counts().sort_index()
                print("  小时分布:")
                for hour in range(24):
                    count = hour_dist.get(hour, 0)
                    percentage = (count / total_count) * 100
                    if count > 0:
                        print(f"    {hour:02d}:00 - {count:>8,} 次 ({percentage:>5.1f}%)")

                # 星期分布
                weekday_dist = converted_timestamps.dt.dayofweek.value_counts().sort_index()
                weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
                print("  星期分布:")
                for day in range(7):
                    count = weekday_dist.get(day, 0)
                    percentage = (count / total_count) * 100
                    day_name = weekday_names[day]
                    print(f"    {day_name} - {count:>8,} 次 ({percentage:>5.1f}%)")

                # 周末占比
                weekend_mask = converted_timestamps.dt.dayofweek.isin([5, 6])
                weekend_count = weekend_mask.sum()
                weekend_percentage = (weekend_count / total_count) * 100
                print(f"  周末占比: {weekend_count:,}/{total_count:,} = {weekend_percentage:.2f}%")

                return {
                    'success': True,
                    'unit': unit,
                    'converted_timestamps': converted_timestamps,
                    'time_range': {'min': min_time, 'max': max_time, 'days': time_span.days},
                    'hour_distribution': hour_dist.to_dict(),
                    'weekday_distribution': weekday_dist.to_dict(),
                    'weekend_percentage': weekend_percentage
                }
            else:
                print("❌ 所有时间戳转换失败")
                return {'success': False, 'error': '所有时间戳转换失败'}

        except Exception as e:
            print(f"❌ 时间戳转换异常: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}