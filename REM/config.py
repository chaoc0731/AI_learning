# config.py
import os


class Config:
    # 模型配置 - 使用本地路径
    MODEL_NAME = "bert-base-chinese"
    LOCAL_MODEL_PATH = r'G:\Pycharm\Project\dataroot\models\bert-base-chinese'

    # 训练参数
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 10

    # 路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "checkpoint"), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 关系类型
    RELATION_TYPES = [
        "治疗", "引起", "预防", "诊断", "副作用", "无关系"
    ]

    @classmethod
    def get_model_path(cls):
        """获取模型路径，优先使用本地模型"""
        # if os.path.exists(cls.LOCAL_MODEL_PATH):
        return cls.LOCAL_MODEL_PATH
        # else:
        #     print(f"警告: 本地模型不存在 {cls.LOCAL_MODEL_PATH}，将在线下载")
        #     return cls.MODEL_NAME