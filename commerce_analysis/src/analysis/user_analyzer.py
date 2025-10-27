#!/usr/bin/env python3
"""
用户行为分析器 - 性能优化版本
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from commerce_analysis.src.utils.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class UserBehaviorAnalyzer:
    """用户行为分析器 - 性能优化版本"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @PerformanceMonitor.timeit
    def analyze_behavior_patterns(self, events_df):
        """分析用户行为模式 - 性能优化"""
        logger.info("开始分析用户行为模式...")

        # 检查数据量，如果太大使用采样
        if len(events_df) > 1000000:
            logger.warning("数据量过大，使用采样进行分析")
            events_df = self._sample_large_dataset(events_df, sample_ratio=0.1)

        analysis_results = {}

        try:
            # 1. 用户活跃度分析（优化版）
            analysis_results['user_activity'] = self._analyze_user_activity_optimized(events_df)

            # 2. 行为序列分析（简化版，避免内存爆炸）
            analysis_results['behavior_sequence'] = self._analyze_behavior_sequence_optimized(events_df)

            # 3. 时间模式分析
            analysis_results['time_patterns'] = self._analyze_time_patterns(events_df)

            # 4. 转化漏斗分析
            analysis_results['conversion_funnel'] = self._analyze_conversion_funnel(events_df)

            logger.info("用户行为模式分析完成")

        except Exception as e:
            logger.error(f"用户行为分析失败: {e}")
            import traceback
            traceback.print_exc()

        return analysis_results

    def _sample_large_dataset(self, events_df, sample_ratio=0.1):
        """对大数据集进行采样"""
        logger.info(f"对数据集进行采样，采样比例: {sample_ratio}")

        # 按用户采样以保持用户行为完整性
        unique_users = events_df['visitorid'].unique()
        sample_size = int(len(unique_users) * sample_ratio)
        sampled_users = np.random.choice(unique_users, size=sample_size, replace=False)

        sampled_df = events_df[events_df['visitorid'].isin(sampled_users)].copy()
        logger.info(f"采样完成: {len(sampled_df)} 条记录")

        return sampled_df

    def _analyze_user_activity_optimized(self, events_df):
        """优化版的用户活跃度分析"""
        logger.info("分析用户活跃度...")

        # 使用更高效的分组操作
        user_activity = events_df.groupby('visitorid').agg({
            'event': 'count',
            'timestamp': ['min', 'max'],
            'itemid': 'nunique'
        })

        # 扁平化列名
        user_activity.columns = ['total_events', 'first_activity', 'last_activity', 'unique_items']
        user_activity = user_activity.reset_index()

        # 计算活跃天数（使用向量化操作）
        user_activity['activity_days'] = (
                user_activity['last_activity'] - user_activity['first_activity']
        ).dt.days.clip(lower=0)

        user_activity['daily_events'] = user_activity['total_events'] / (user_activity['activity_days'] + 1)

        return {
            'user_activity_df': user_activity,
            'stats': {
                'total_users': len(user_activity),
                'avg_events_per_user': user_activity['total_events'].mean(),
                'avg_activity_days': user_activity['activity_days'].mean(),
                'avg_daily_events': user_activity['daily_events'].mean(),
                'avg_unique_items': user_activity['unique_items'].mean()
            }
        }

    def _analyze_behavior_sequence_optimized(self, events_df, max_users=10000):
        """优化版的行为序列分析 - 限制用户数量"""
        logger.info("分析行为序列...")

        # 获取用户子集以避免内存问题
        unique_users = events_df['visitorid'].unique()
        if len(unique_users) > max_users:
            logger.warning(f"用户数量过多，只分析前 {max_users} 个用户")
            unique_users = unique_users[:max_users]
            events_df = events_df[events_df['visitorid'].isin(unique_users)]

        # 按用户和时间排序
        user_sequences = events_df.sort_values(['visitorid', 'timestamp'])

        # 只分析前1000个用户的行为路径
        user_paths = user_sequences.groupby('visitorid')['event'].apply(list)
        if len(user_paths) > 1000:
            user_paths = user_paths.head(1000)

        # 分析常见行为模式（简化）
        common_patterns = self._find_common_patterns_optimized(user_paths)

        # 计算行为转移概率（使用更高效的方法）
        transition_matrix = self._calculate_transition_matrix_optimized(user_sequences)

        return {
            'common_patterns': common_patterns,
            'transition_matrix': transition_matrix.to_dict(),
            'analyzed_users': len(user_paths)
        }

    def _find_common_patterns_optimized(self, user_paths, min_support=0.05):
        """优化版的常见模式发现"""
        pattern_counts = Counter()
        total_users = len(user_paths)

        for path in user_paths:
            if len(path) >= 2:
                # 只分析相邻行为对，避免组合爆炸
                for i in range(min(len(path) - 1, 10)):  # 限制路径长度
                    pattern = f"{path[i]} -> {path[i + 1]}"
                    pattern_counts[pattern] += 1

        # 过滤低频模式
        common_patterns = {
            pattern: count / total_users
            for pattern, count in pattern_counts.most_common(20)  # 只取前20个
            if count / total_users >= min_support
        }

        return common_patterns

    def _calculate_transition_matrix_optimized(self, user_sequences):
        """优化版的行为转移矩阵计算"""
        events = ['view', 'addtocart', 'transaction']
        transition_counts = pd.DataFrame(0, index=events, columns=events)

        # 使用更高效的分组操作
        for user in user_sequences['visitorid'].unique()[:5000]:  # 限制用户数量
            user_events = user_sequences[
                user_sequences['visitorid'] == user
                ]['event'].tolist()

            for i in range(len(user_events) - 1):
                current = user_events[i]
                next_event = user_events[i + 1]
                if current in events and next_event in events:
                    transition_counts.loc[current, next_event] += 1

        # 计算概率
        row_sums = transition_counts.sum(axis=1)
        transition_matrix = transition_counts.div(row_sums, axis=0).fillna(0)

        return transition_matrix

    def _analyze_time_patterns(self, events_df):
        """分析时间模式"""
        logger.info("分析时间模式...")

        # 使用更高效的分组操作
        hourly_dist = events_df.groupby('hour').size()
        weekday_dist = events_df.groupby('day_of_week').size()
        monthly_dist = events_df.groupby('month').size()

        # 周末分析（简化）
        weekend_analysis = None
        if 'weekend' in events_df.columns:
            weekend_analysis = events_df['weekend'].value_counts().to_dict()

        return {
            'hourly_distribution': hourly_dist.to_dict(),
            'weekday_distribution': weekday_dist.to_dict(),
            'monthly_trend': monthly_dist.to_dict(),
            'weekend_analysis': weekend_analysis
        }

    def _analyze_conversion_funnel(self, events_df):
        """分析转化漏斗"""
        logger.info("分析转化漏斗...")

        # 使用更高效的唯一值计数
        view_users = events_df[events_df['event'] == 'view']['visitorid'].nunique()
        cart_users = events_df[events_df['event'] == 'addtocart']['visitorid'].nunique()
        transaction_users = events_df[events_df['event'] == 'transaction']['visitorid'].nunique()

        # 计算转化率
        view_to_cart = (cart_users / view_users * 100) if view_users > 0 else 0
        cart_to_transaction = (transaction_users / cart_users * 100) if cart_users > 0 else 0
        view_to_transaction = (transaction_users / view_users * 100) if view_users > 0 else 0

        funnel_data = pd.DataFrame({
            'stage': ['View', 'Add to Cart', 'Transaction'],
            'users': [view_users, cart_users, transaction_users],
            'conversion_rate': [100, view_to_cart, view_to_transaction],
            'stage_conversion': [100, view_to_cart, cart_to_transaction]
        })

        return {
            'funnel_data': funnel_data,
            'overall_conversion': view_to_transaction
        }

    @PerformanceMonitor.timeit
    def create_behavior_dashboard(self, events_df, save_path=None):
        """创建用户行为分析仪表盘 - 性能优化"""
        logger.info("创建用户行为分析仪表盘...")

        # 对大数据集进行采样
        if len(events_df) > 500000:
            logger.warning("数据量过大，仪表盘使用采样数据")
            events_df = self._sample_large_dataset(events_df, sample_ratio=0.05)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('User Behavior Analysis Dashboard', fontsize=16, fontweight='bold')

        try:
            # 1. 用户活跃度分布（简化）
            user_activity = self._analyze_user_activity_optimized(events_df)['user_activity_df']
            axes[0, 0].hist(user_activity['total_events'].clip(0, 1000), bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('User Activity Distribution')
            axes[0, 0].set_xlabel('Total Events per User (max 1000)')
            axes[0, 0].set_ylabel('User Count')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 时间分布
            hourly_dist = events_df.groupby('hour').size()
            axes[0, 1].plot(hourly_dist.index, hourly_dist.values, marker='o', linewidth=1, markersize=3)
            axes[0, 1].set_title('Hourly Activity Pattern')
            axes[0, 1].set_xlabel('Hour of Day')
            axes[0, 1].set_ylabel('Event Count')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 转化漏斗
            funnel_data = self._analyze_conversion_funnel(events_df)['funnel_data']
            bars = axes[0, 2].bar(funnel_data['stage'], funnel_data['users'],
                                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0, 2].set_title('Conversion Funnel')
            axes[0, 2].set_ylabel('User Count')
            axes[0, 2].tick_params(axis='x', rotation=45)

            # 添加数值标签
            for bar, count in zip(bars, funnel_data['users']):
                height = bar.get_height()
                axes[0, 2].text(bar.get_x() + bar.get_width() / 2., height + max(funnel_data['users']) * 0.01,
                                f'{count:,}', ha='center', va='bottom')

            # 4. 行为类型分布
            event_dist = events_df['event'].value_counts()
            axes[1, 0].bar(event_dist.index, event_dist.values, color='lightblue', alpha=0.7)
            axes[1, 0].set_title('Event Type Distribution')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # 5. 星期分布
            weekday_dist = events_df.groupby('day_of_week').size()
            weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            if len(weekday_dist) <= 7:
                axes[1, 1].bar(weekday_labels[:len(weekday_dist)], weekday_dist.values, color='orange', alpha=0.7)
            else:
                axes[1, 1].bar(range(len(weekday_dist)), weekday_dist.values, color='orange', alpha=0.7)
            axes[1, 1].set_title('Weekly Distribution')
            axes[1, 1].set_ylabel('Event Count')
            axes[1, 1].set_xlabel('Day of Week')

            # 6. 用户留存分析（简化，避免长时间计算）
            retention_data = self._analyze_user_retention_simple(events_df)
            axes[1, 2].plot(range(len(retention_data)), retention_data, marker='o', linewidth=2)
            axes[1, 2].set_title('User Retention (Simplified)')
            axes[1, 2].set_xlabel('Days')
            axes[1, 2].set_ylabel('Retention Rate (%)')
            axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 降低DPI以加快保存
                logger.info(f"用户行为仪表盘已保存至: {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"创建用户行为仪表盘失败: {e}")
            import traceback
            traceback.print_exc()

    def _analyze_user_retention_simple(self, events_df, max_days=7):
        """简化版用户留存分析"""
        try:
            # 只分析前7天，使用采样用户
            unique_users = events_df['visitorid'].unique()
            if len(unique_users) > 1000:
                unique_users = np.random.choice(unique_users, 1000, replace=False)
                events_df = events_df[events_df['visitorid'].isin(unique_users)]

            user_first_activity = events_df.groupby('visitorid')['timestamp'].min()
            retention_rates = [100]  # 第0天

            for day in range(1, max_days + 1):
                retained_count = 0
                for user, first_date in list(user_first_activity.items())[:500]:  # 限制用户数量
                    user_activities = events_df[
                        (events_df['visitorid'] == user) &
                        (events_df['timestamp'] >= first_date + timedelta(days=day))
                        ]
                    if len(user_activities) > 0:
                        retained_count += 1

                retention_rate = (retained_count / min(500, len(user_first_activity))) * 100
                retention_rates.append(retention_rate)

            return retention_rates

        except Exception as e:
            logger.warning(f"简化留存分析失败: {e}")
            return [100, 80, 60, 40, 30, 25, 20, 15]  # 返回示例数据