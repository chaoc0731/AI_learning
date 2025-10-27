#!/usr/bin/env python3
"""
商品分析器
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

logger = logging.getLogger(__name__)


class ProductAnalyzer:
    """商品分析器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_products(self, events_df, item_properties_df):
        """综合分析商品表现"""
        logger.info("开始商品分析...")

        analysis_results = {}

        try:
            # 1. 商品行为分析
            analysis_results['product_behavior'] = self._analyze_product_behavior(events_df)

            # 2. 热门商品识别
            analysis_results['popular_products'] = self._identify_popular_products(events_df)

            # 3. 商品转化率分析
            analysis_results['conversion_analysis'] = self._analyze_product_conversion(events_df)

            # 4. 商品属性分析
            if not item_properties_df.empty:
                analysis_results['property_analysis'] = self._analyze_product_properties(item_properties_df)

            # 5. 商品关联分析
            analysis_results['association_analysis'] = self._analyze_product_associations(events_df)

            logger.info("商品分析完成")

        except Exception as e:
            logger.error(f"商品分析失败: {e}")

        return analysis_results

    def _analyze_product_behavior(self, events_df):
        """分析商品行为数据"""
        product_behavior = events_df.groupby('itemid').agg({
            'event': 'count',
            'visitorid': 'nunique',
            'timestamp': ['min', 'max']
        }).reset_index()

        product_behavior.columns = ['itemid', 'total_events', 'unique_users', 'first_seen', 'last_seen']

        # 计算平均每个用户的行为数
        product_behavior['events_per_user'] = product_behavior['total_events'] / product_behavior['unique_users']

        # 计算活跃天数
        product_behavior['active_days'] = (product_behavior['last_seen'] - product_behavior['first_seen']).dt.days
        product_behavior['active_days'] = product_behavior['active_days'].clip(lower=0)

        return product_behavior

    def _identify_popular_products(self, events_df, top_n=20):
        """识别热门商品"""
        # 按不同指标排名
        popularity_metrics = {}

        # 总行为数排名
        total_events_rank = events_df.groupby('itemid').size().sort_values(ascending=False).head(top_n)
        popularity_metrics['by_total_events'] = total_events_rank.to_dict()

        # 独立用户数排名
        unique_users_rank = events_df.groupby('itemid')['visitorid'].nunique().sort_values(ascending=False).head(top_n)
        popularity_metrics['by_unique_users'] = unique_users_rank.to_dict()

        # 交易次数排名
        transaction_rank = events_df[events_df['event'] == 'transaction'].groupby('itemid').size().sort_values(
            ascending=False).head(top_n)
        popularity_metrics['by_transactions'] = transaction_rank.to_dict()

        # 综合评分（加权排名）
        product_scores = self._calculate_product_scores(events_df)
        popularity_metrics['by_composite_score'] = dict(
            sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:top_n])

        return popularity_metrics

    def _calculate_product_scores(self, events_df):
        """计算商品综合评分"""
        # 计算基础指标
        product_stats = events_df.groupby('itemid').agg({
            'event': 'count',
            'visitorid': 'nunique'
        }).rename(columns={'event': 'total_events', 'visitorid': 'unique_users'})

        # 计算交易相关指标
        transaction_stats = events_df[events_df['event'] == 'transaction'].groupby('itemid').agg({
            'event': 'count',
            'visitorid': 'nunique'
        }).rename(columns={'event': 'transaction_count', 'visitorid': 'transaction_users'})

        # 合并数据
        product_scores = product_stats.merge(transaction_stats, on='itemid', how='left').fillna(0)

        # 计算转化率
        product_scores['view_to_transaction_rate'] = product_scores['transaction_count'] / product_scores[
            'total_events']

        # 标准化指标
        for col in ['total_events', 'unique_users', 'transaction_count', 'transaction_users',
                    'view_to_transaction_rate']:
            if product_scores[col].max() > 0:
                product_scores[f'{col}_normalized'] = product_scores[col] / product_scores[col].max()
            else:
                product_scores[f'{col}_normalized'] = 0

        # 计算综合评分（加权平均）
        weights = {
            'total_events_normalized': 0.2,
            'unique_users_normalized': 0.3,
            'transaction_count_normalized': 0.25,
            'transaction_users_normalized': 0.15,
            'view_to_transaction_rate_normalized': 0.1
        }

        product_scores['composite_score'] = 0
        for col, weight in weights.items():
            product_scores['composite_score'] += product_scores[col] * weight

        return product_scores['composite_score'].to_dict()

    def _analyze_product_conversion(self, events_df):
        """分析商品转化率"""
        # 计算每个商品的转化漏斗
        product_funnel = events_df.groupby(['itemid', 'event']).size().unstack(fill_value=0)

        # 确保所有事件类型都存在
        for event_type in ['view', 'addtocart', 'transaction']:
            if event_type not in product_funnel.columns:
                product_funnel[event_type] = 0

        # 计算转化率
        product_funnel['view_to_cart_rate'] = product_funnel['addtocart'] / product_funnel['view']
        product_funnel['view_to_transaction_rate'] = product_funnel['transaction'] / product_funnel['view']
        product_funnel['cart_to_transaction_rate'] = product_funnel['transaction'] / product_funnel['addtocart']

        # 处理除零错误
        product_funnel = product_funnel.replace([np.inf, -np.inf], 0).fillna(0)

        return product_funnel

    def _analyze_product_properties(self, item_properties_df):
        """分析商品属性"""
        if item_properties_df.empty:
            return {}

        # 分析属性分布
        property_distribution = item_properties_df['property'].value_counts().head(20)

        # 分析属性值分布
        property_value_analysis = {}
        for property_name in item_properties_df['property'].unique()[:10]:  # 只分析前10个属性
            property_values = item_properties_df[item_properties_df['property'] == property_name][
                'value'].value_counts().head(10)
            property_value_analysis[property_name] = property_values.to_dict()

        return {
            'property_distribution': property_distribution.to_dict(),
            'property_value_analysis': property_value_analysis
        }

    def _analyze_product_associations(self, events_df, min_support=0.001):
        """分析商品关联关系"""
        try:
            # 获取用户的商品交互序列
            user_products = events_df.groupby('visitorid')['itemid'].apply(list)

            # 分析商品共现
            co_occurrence = self._calculate_co_occurrence(user_products)

            # 分析购买关联
            purchase_pairs = self._analyze_purchase_pairs(events_df)

            return {
                'co_occurrence': dict(sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:20]),
                'purchase_pairs': dict(sorted(purchase_pairs.items(), key=lambda x: x[1], reverse=True)[:20])
            }

        except Exception as e:
            logger.error(f"商品关联分析失败: {e}")
            return {}

    def _calculate_co_occurrence(self, user_products):
        """计算商品共现频率"""
        co_occurrence = Counter()

        for products in user_products:
            unique_products = list(set(products))  # 去重
            for i in range(len(unique_products)):
                for j in range(i + 1, len(unique_products)):
                    pair = tuple(sorted([unique_products[i], unique_products[j]]))
                    co_occurrence[pair] += 1

        return co_occurrence

    def _analyze_purchase_pairs(self, events_df):
        """分析购买商品对"""
        # 获取有交易行为的用户
        transaction_users = events_df[events_df['event'] == 'transaction']['visitorid'].unique()

        purchase_pairs = Counter()

        for user in transaction_users[:1000]:  # 限制用户数量避免内存问题
            user_transactions = events_df[
                (events_df['visitorid'] == user) &
                (events_df['event'] == 'transaction')
                ]['itemid'].unique()

            if len(user_transactions) >= 2:
                for i in range(len(user_transactions)):
                    for j in range(i + 1, len(user_transactions)):
                        pair = tuple(sorted([user_transactions[i], user_transactions[j]]))
                        purchase_pairs[pair] += 1

        return purchase_pairs

    def create_product_dashboard(self, events_df, item_properties_df, save_path=None):
        """创建商品分析仪表盘"""
        logger.info("创建商品分析仪表盘...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Product Analysis Dashboard', fontsize=16, fontweight='bold')

        try:
            # 1. 热门商品（按总行为数）
            top_products = events_df.groupby('itemid').size().sort_values(ascending=False).head(10)
            axes[0, 0].bar(range(len(top_products)), top_products.values, color='skyblue')
            axes[0, 0].set_title('Top 10 Products by Total Events')
            axes[0, 0].set_ylabel('Event Count')
            axes[0, 0].set_xticks(range(len(top_products)))
            axes[0, 0].set_xticklabels([str(x) for x in top_products.index], rotation=45)

            # 2. 商品转化率分布
            conversion_analysis = self._analyze_product_conversion(events_df)
            axes[0, 1].hist(conversion_analysis['view_to_transaction_rate'].clip(0, 1),
                            bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Product Conversion Rate Distribution')
            axes[0, 1].set_xlabel('View to Transaction Rate')
            axes[0, 1].set_ylabel('Product Count')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 商品活跃度分布
            product_behavior = self._analyze_product_behavior(events_df)
            axes[0, 2].hist(product_behavior['unique_users'], bins=50, alpha=0.7, edgecolor='black', color='green')
            axes[0, 2].set_title('Product Popularity Distribution')
            axes[0, 2].set_xlabel('Unique Users per Product')
            axes[0, 2].set_ylabel('Product Count')
            axes[0, 2].grid(True, alpha=0.3)

            # 4. 事件类型分布（热门商品）
            top_product_ids = top_products.index[:5]
            for product_id in top_product_ids:
                product_events = events_df[events_df['itemid'] == product_id]['event'].value_counts()
                axes[1, 0].bar(product_events.index, product_events.values, alpha=0.7, label=str(product_id))
            axes[1, 0].set_title('Event Distribution for Top Products')
            axes[1, 0].set_ylabel('Event Count')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)

            # 5. 商品属性分析（如果有属性数据）
            if not item_properties_df.empty:
                top_properties = item_properties_df['property'].value_counts().head(10)
                axes[1, 1].bar(range(len(top_properties)), top_properties.values, color='orange')
                axes[1, 1].set_title('Top 10 Product Properties')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].set_xticks(range(len(top_properties)))
                axes[1, 1].set_xticklabels(top_properties.index, rotation=45)

            # 6. 时间趋势（热门商品）
            for product_id in top_product_ids[:3]:
                product_trend = events_df[events_df['itemid'] == product_id].groupby('date').size()
                axes[1, 2].plot(product_trend.index, product_trend.values, label=str(product_id), marker='o',
                                markersize=3)
            axes[1, 2].set_title('Popular Products Trend Over Time')
            axes[1, 2].set_xlabel('Date')
            axes[1, 2].set_ylabel('Daily Events')
            axes[1, 2].legend()
            axes[1, 2].tick_params(axis='x', rotation=45)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"商品分析仪表盘已保存至: {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"创建商品分析仪表盘失败: {e}")