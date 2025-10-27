import pandas as pd
import logging
from pathlib import Path
from commerce_analysis.config.setting import Config


logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器 - 专门验证多文件数据集的完整性"""

    def __init__(self):
        self.config = Config

    def validate_item_properties_parts(self):
        """验证多个item_properties文件的完整性"""
        logger.info("开始验证item_properties文件完整性...")

        property_parts = ['item_properties_part1', 'item_properties_part2']
        validation_results = {}

        for part_name in property_parts:
            file_path = self.config.RAW_DATA_DIR / self.config.DATA_FILES[part_name]

            if not file_path.exists():
                logger.warning(f"文件不存在: {file_path}")
                validation_results[part_name] = {'exists': False}
                continue

            try:
                # 快速读取前几行验证文件格式
                sample_df = pd.read_csv(file_path, nrows=1000)
                validation_results[part_name] = {
                    'exists': True,
                    'shape': sample_df.shape,
                    'columns': list(sample_df.columns),
                    'sample_size': len(sample_df)
                }

                logger.info(f"{part_name} 验证通过: {sample_df.shape}")

            except Exception as e:
                logger.error(f"{part_name} 验证失败: {e}")
                validation_results[part_name] = {'exists': True, 'error': str(e)}

        return validation_results

    def check_data_overlap(self):
        """检查多个item_properties文件之间的数据重叠"""
        logger.info("检查item_properties文件间的数据重叠...")

        try:
            # 读取两个文件的第一部分进行比较
            part1_path = self.config.RAW_DATA_DIR / self.config.DATA_FILES['item_properties_part1']
            part2_path = self.config.RAW_DATA_DIR / self.config.DATA_FILES['item_properties_part2']

            part1_sample = pd.read_csv(part1_path, nrows=10000)
            part2_sample = pd.read_csv(part2_path, nrows=10000)

            # 检查列是否一致
            columns_match = set(part1_sample.columns) == set(part2_sample.columns)

            # 检查itemid是否有重叠
            part1_items = set(part1_sample['itemid'].unique())
            part2_items = set(part2_sample['itemid'].unique())
            overlap_items = part1_items.intersection(part2_items)
            overlap_ratio = len(overlap_items) / len(part1_items.union(part2_items)) if part1_items.union(
                part2_items) else 0

            overlap_info = {
                'columns_match': columns_match,
                'part1_unique_items': len(part1_items),
                'part2_unique_items': len(part2_items),
                'overlap_items': len(overlap_items),
                'overlap_ratio': overlap_ratio
            }

            logger.info(f"数据重叠分析: {overlap_info}")
            return overlap_info

        except Exception as e:
            logger.error(f"数据重叠检查失败: {e}")
            return None


if __name__ == "__main__":
    # 快速验证数据文件
    validator = DataValidator()

    print("=== 数据文件验证 ===")
    part_validation = validator.validate_item_properties_parts()
    for part, result in part_validation.items():
        print(f"{part}: {result}")

    print("\n=== 数据重叠检查 ===")
    overlap_info = validator.check_data_overlap()
    print(overlap_info)