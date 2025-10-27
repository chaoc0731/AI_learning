# config/settings.py
import os
from pathlib import Path

# 修复路径问题 - 使用绝对路径
BASE_DIR = Path(__file__).parent.parent


class Config:
    """项目配置类"""

    # 路径配置 - 使用绝对路径
    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = DATA_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "outputs"

    # 数据文件配置
    DATA_FILES = {
        'events': 'events.csv',
        'item_properties_part1': 'item_properties_part1.csv',
        'item_properties_part2': 'item_properties_part2.csv',
        'category_tree': 'category_tree.csv'
    }

    # 分析参数
    RFM_SEGMENT_CONFIG = {
        'high_value': 10,
        'potential': 7,
        'general': 5
    }

    EVENT_WEIGHTS = {
        'view': 1,
        'addtocart': 5,
        'transaction': 10
    }

    # 数据处理参数
    ITEM_PROPERTIES_CHUNK_SIZE = 100000

    # 可视化配置
    PLOT_STYLE = 'default'
    COLOR_PALETTE = 'husl'

    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR,
            cls.OUTPUT_DIR / "figures",
            cls.OUTPUT_DIR / "reports",
            cls.OUTPUT_DIR / "predictions"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)