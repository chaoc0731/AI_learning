import pandas as pd
import numpy as np
from datetime import datetime
import logging
from commerce_analysis.config.setting import Config


logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器 - 优化处理多文件合并的数据"""

    def __init__(self):
        self.config = Config

    def preprocess_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """预处理事件数据 - 修复时间戳转换"""
        logger.info("开始预处理events数据...")

        # 复制数据避免修改原始数据
        df = events_df.copy()

        # 时间戳转换 - 专门处理毫秒时间戳
        logger.info("处理时间戳数据...")

        # 检查timestamp列是否存在
        if 'timestamp' not in df.columns:
            logger.error("数据中缺少timestamp列")
            # 尝试查找可能的时间列
            time_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if time_columns:
                logger.info(f"找到可能的时间列: {time_columns}")
                df = df.rename(columns={time_columns[0]: 'timestamp'})
            else:
                logger.error("未找到时间列，无法进行时间分析")
                return df

        # 转换时间戳 - 专门处理毫秒时间戳
        try:
            logger.info("检测到int64时间戳，按毫秒时间戳处理...")

            # 检查时间戳范围，确定是秒还是毫秒
            sample_timestamp = df['timestamp'].iloc[0]
            logger.info(f"示例时间戳: {sample_timestamp}")

            # 判断时间戳单位（秒还是毫秒）
            # 通常毫秒时间戳是13位数字，秒时间戳是10位数字
            if sample_timestamp > 1e12:  # 大于 10^12，很可能是毫秒
                logger.info("时间戳为毫秒单位")
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                logger.info("时间戳为秒单位")
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            # 检查转换结果
            valid_timestamps = df['timestamp'].notna().sum()
            invalid_timestamps = df['timestamp'].isna().sum()

            logger.info(f"时间戳转换结果: {valid_timestamps} 有效, {invalid_timestamps} 无效")

            if valid_timestamps == 0:
                logger.error("所有时间戳转换失败")
                # 尝试其他转换方法
                logger.info("尝试其他转换方法...")
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            # 显示转换后的时间范围
            if valid_timestamps > 0:
                time_min = df['timestamp'].min()
                time_max = df['timestamp'].max()
                time_span = time_max - time_min
                logger.info(f"时间范围: {time_min} 到 {time_max}")
                logger.info(f"时间跨度: {time_span.days} 天")

                # 检查时间是否合理（应该在合理的历史范围内）
                current_year = pd.Timestamp.now().year
                data_year = time_min.year
                if data_year < 2000 or data_year > current_year:
                    logger.warning(f"时间数据年份异常: {data_year}")

        except Exception as e:
            logger.error(f"时间戳转换失败: {e}")
            import traceback
            traceback.print_exc()
            return df

        # 提取时间特征
        logger.info("提取时间特征...")
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 周一=0, 周日=6
        df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # 周六=5, 周日=6
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        df['year'] = df['timestamp'].dt.year

        # 添加事件权重
        df['event_weight'] = df['event'].map(self.config.EVENT_WEIGHTS)

        # 记录时间特征统计
        logger.info("时间特征统计:")
        logger.info(f"  时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        logger.info(f"  数据年份: {df['year'].unique()}")
        logger.info(f"  周末行为占比: {df['weekend'].mean() * 100:.2f}%")

        # 小时分布统计
        hour_dist = df['hour'].value_counts().sort_index()
        peak_hour = hour_dist.idxmax() if not hour_dist.empty else -1
        logger.info(f"  最活跃时段: {peak_hour}:00")
        logger.info(f"  小时分布: {dict(hour_dist.head())}...")

        # 星期分布统计
        weekday_dist = df['day_of_week'].value_counts().sort_index()
        logger.info(f"  星期分布: {dict(weekday_dist)}")

        logger.info(f"events数据预处理完成，形状: {df.shape}")
        return df

    def preprocess_item_properties(self, item_properties_df: pd.DataFrame) -> pd.DataFrame:
        """预处理商品属性数据 - 优化处理合并后的大文件"""
        logger.info("开始预处理item_properties数据...")

        if item_properties_df.empty:
            logger.warning("item_properties数据为空，返回空DataFrame")
            return pd.DataFrame()

        df = item_properties_df.copy()

        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 数据质量检查
        initial_count = len(df)
        df = df.dropna(subset=['itemid', 'property'])
        after_clean_count = len(df)

        if initial_count != after_clean_count:
            logger.warning(f"清理了 {initial_count - after_clean_count} 条包含空值的记录")

        # 按商品和时间排序，获取每个商品的最新属性
        logger.info("正在提取每个商品的最新属性...")
        latest_properties = (df.sort_values(['itemid', 'timestamp'], ascending=[True, False])
                             .groupby('itemid')
                             .head(1)  # 获取每个商品的最新记录
                             .reset_index(drop=True))

        logger.info(f"item_properties数据预处理完成，提取了 {len(latest_properties)} 个商品的属性")
        return latest_properties

    def optimize_item_properties_memory(self, item_properties_df: pd.DataFrame) -> pd.DataFrame:
        """
        优化item_properties内存使用
        处理大文件时特别有用
        """
        if item_properties_df.empty:
            return item_properties_df

        logger.info("开始优化item_properties内存使用...")

        df = item_properties_df.copy()

        # 优化数据类型
        memory_before = df.memory_usage(deep=True).sum() / 1024 ** 2  # MB

        # 优化数值列
        if 'itemid' in df.columns:
            df['itemid'] = pd.to_numeric(df['itemid'], downcast='integer', errors='coerce')

        # 优化分类列
        categorical_columns = ['property', 'value']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')

        memory_after = df.memory_usage(deep=True).sum() / 1024 ** 2  # MB
        memory_saved = memory_before - memory_after

        logger.info(f"内存优化完成: {memory_before:.2f}MB -> {memory_after:.2f}MB (节省 {memory_saved:.2f}MB)")

        return df

    def merge_datasets(self, events_df: pd.DataFrame,
                       item_properties_df: pd.DataFrame,
                       category_tree_df: pd.DataFrame) -> pd.DataFrame:
        """合并所有数据集 - 优化大文件合并性能"""
        logger.info("开始合并数据集...")

        # 如果item_properties数据很大，使用优化合并策略
        if len(item_properties_df) > 100000:  # 如果数据量很大
            logger.info("检测到大文件，使用优化合并策略...")

            # 首先只合并必要的列
            essential_properties = item_properties_df[['itemid', 'property', 'value']].drop_duplicates()

            # 使用更高效的内存数据类型
            essential_properties = self.optimize_item_properties_memory(essential_properties)

            # 合并数据
            merged_df = events_df.merge(
                essential_properties,
                on='itemid',
                how='left'
            )
        else:
            # 常规合并
            merged_df = events_df.merge(
                item_properties_df[['itemid', 'property', 'value']],
                on='itemid',
                how='left'
            )

        # 如果有分类树数据，也进行合并
        if not category_tree_df.empty:
            merged_df = merged_df.merge(
                category_tree_df,
                left_on='itemid',
                right_on='categoryid',
                how='left'
            )

        logger.info(f"数据集合并完成，最终形状: {merged_df.shape}")

        # 记录合并后的数据分布
        properties_coverage = merged_df['property'].notna().sum() / len(merged_df) * 100
        logger.info(f"属性数据覆盖度: {properties_coverage:.2f}%")

        return merged_df

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """保存处理后的数据 - 支持大文件分块保存"""
        file_path = self.config.PROCESSED_DATA_DIR / filename

        # 如果文件很大，使用parquet格式保存（更好的压缩和性能）
        if len(df) > 100000:
            file_path = file_path.with_suffix('.parquet')
            df.to_parquet(file_path, index=False, compression='snappy')
        else:
            file_path = file_path.with_suffix('.csv')
            df.to_csv(file_path, index=False)

        logger.info(f"处理后的数据已保存至: {file_path}")

    def create_sample_dataset(self, full_df: pd.DataFrame, sample_ratio: float = 0.1) -> pd.DataFrame:
        """
        创建数据样本，用于快速开发和测试
        """
        logger.info(f"创建数据样本，采样比例: {sample_ratio:.1%}")

        # 按用户采样以保持用户行为的完整性
        unique_users = full_df['visitorid'].unique()
        sample_users = np.random.choice(unique_users,
                                        size=int(len(unique_users) * sample_ratio),
                                        replace=False)

        sample_df = full_df[full_df['visitorid'].isin(sample_users)].copy()

        logger.info(f"样本数据创建完成: {len(sample_df):,} 条记录")
        return sample_df




