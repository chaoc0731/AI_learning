import pandas as pd
import logging
from pathlib import Path
from commerce_analysis.config.setting import Config


logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器 - 适配多文件item_properties"""

    def __init__(self):
        self.config = Config
        self.config.create_directories()

    def load_events_data(self) -> pd.DataFrame:
        """加载事件数据"""
        try:
            file_path = self.config.RAW_DATA_DIR / self.config.DATA_FILES['events']
            df = pd.read_csv(file_path)
            logger.info(f"成功加载events数据，形状: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"未找到events数据文件: {file_path}")
            raise

    def load_item_properties_parts(self) -> pd.DataFrame:
        """
        加载多个item_properties文件并合并
        使用分块读取处理大文件
        """
        all_parts = []

        # 定义要加载的文件部分
        property_parts = ['item_properties_part1', 'item_properties_part2']

        for part_name in property_parts:
            try:
                file_path = self.config.RAW_DATA_DIR / self.config.DATA_FILES[part_name]
                logger.info(f"正在加载 {part_name}...")

                # 使用分块读取处理大文件
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=self.config.ITEM_PROPERTIES_CHUNK_SIZE):
                    chunks.append(chunk)

                part_df = pd.concat(chunks, ignore_index=True)
                all_parts.append(part_df)
                logger.info(f"成功加载 {part_name}，形状: {part_df.shape}")

            except FileNotFoundError:
                logger.warning(f"未找到 {part_name} 数据文件: {file_path}")
                continue

        if not all_parts:
            logger.error("未找到任何item_properties文件")
            return pd.DataFrame()

        # 合并所有部分
        combined_df = pd.concat(all_parts, ignore_index=True)
        logger.info(f"所有item_properties文件合并完成，总形状: {combined_df.shape}")

        return combined_df

    def load_category_tree(self) -> pd.DataFrame:
        """加载分类树数据"""
        try:
            file_path = self.config.RAW_DATA_DIR / self.config.DATA_FILES['category_tree']
            df = pd.read_csv(file_path)
            logger.info(f"成功加载category_tree数据，形状: {df.shape}")
            return df
        except FileNotFoundError:
            logger.warning(f"未找到category_tree数据文件: {file_path}")
            return pd.DataFrame()

    def load_all_data(self):
        """加载所有数据"""
        events_df = self.load_events_data()
        item_properties_df = self.load_item_properties_parts()  # 修改为新的加载方法
        category_tree_df = self.load_category_tree()

        return events_df, item_properties_df, category_tree_df

    def validate_data_quality(self, events_df, item_properties_df, category_tree_df):
        """验证数据质量"""
        logger.info("开始数据质量验证...")

        # 检查events数据
        if events_df.empty:
            logger.error("events数据为空")
            return False

        # 检查item_properties数据
        if item_properties_df.empty:
            logger.warning("item_properties数据为空")

        # 检查必要列是否存在
        required_events_columns = ['timestamp', 'visitorid', 'event', 'itemid']
        missing_columns = [col for col in required_events_columns if col not in events_df.columns]
        if missing_columns:
            logger.error(f"events数据缺少必要列: {missing_columns}")
            return False

        # 检查数据完整性
        logger.info(f"events数据记录数: {len(events_df):,}")
        logger.info(f"item_properties数据记录数: {len(item_properties_df):,}")
        logger.info(f"category_tree数据记录数: {len(category_tree_df):,}")

        return True