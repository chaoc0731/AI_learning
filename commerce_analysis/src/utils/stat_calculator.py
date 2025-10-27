import logger
import pandas as pd


def calculate_time_patterns(events_df):
    """计算时间模式 - 返回完整的时间统计信息"""
    try:
        # 确保时间戳列存在且是datetime类型
        if 'timestamp' not in events_df.columns or not pd.api.types.is_datetime64_any_dtype(events_df['timestamp']):
            logger.error("时间数据不可用")
            return {'peak_hour': 0, 'weekend_ratio': 0}

        # 重新提取时间特征
        events_df = events_df.copy()
        events_df['hour'] = events_df['timestamp'].dt.hour
        events_df['day_of_week'] = events_df['timestamp'].dt.dayofweek
        events_df['weekend'] = events_df['day_of_week'].isin([5, 6]).astype(int)

        # 1. 小时分布分析
        hourly_dist = events_df['hour'].value_counts().sort_index()

        # 找到最活跃时段
        if not hourly_dist.empty:
            peak_hour = hourly_dist.idxmax()
            peak_count = hourly_dist.max()
            total_events = len(events_df)
            peak_percentage = (peak_count / total_events) * 100

            logger.info(f"最活跃时段分析: {peak_hour}:00 ({peak_count}次, {peak_percentage:.1f}%)")
        else:
            peak_hour = 0
            peak_count = 0
            peak_percentage = 0

        # 2. 周末分析
        weekend_count = events_df['weekend'].sum()
        total_count = len(events_df)
        weekend_ratio = (weekend_count / total_count * 100) if total_count > 0 else 0

        logger.info(f"周末行为统计: {weekend_count}/{total_count} = {weekend_ratio:.2f}%")

        # 3. 详细的时间分布
        hourly_distribution = hourly_dist.to_dict()
        weekday_distribution = events_df['day_of_week'].value_counts().sort_index().to_dict()

        # 转换为更易读的格式
        readable_hourly = {f"{hour:02d}:00": count for hour, count in hourly_distribution.items()}

        weekday_names = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
        readable_weekday = {weekday_names.get(day, f'周{day + 1}'): count for day, count in
                            weekday_distribution.items()}

        return {
            'peak_hour': peak_hour,
            'weekend_ratio': weekend_ratio,
            'hourly_distribution': readable_hourly,
            'weekday_distribution': readable_weekday,
            'peak_hour_details': {
                'hour': peak_hour,
                'count': peak_count,
                'percentage': peak_percentage
            },
            'total_events': total_count  # 添加总事件数用于计算百分比
        }

    except Exception as e:
        logger.error(f"计算时间模式失败: {e}")
        return {
            'peak_hour': 0,
            'weekend_ratio': 0,
            'hourly_distribution': {},
            'weekday_distribution': {},
            'peak_hour_details': {
                'hour': 0,
                'count': 0,
                'percentage': 0
            },
            'total_events': 0,
            'error': str(e)
        }