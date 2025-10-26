# 项目配置参数
class Config:
    # 数据路径
    RAW_DATA_PATH = r'G:\Pycharm\Project\AI_learning\20231118\NLPProject05\Predictive_Maintenance\data\raw\predictive_maintenance.csv'
    PROCESSED_DATA_PATH = 'data/processed/processed_data.csv'

    # 模型参数
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2

    # 特征列名
    FEATURE_COLUMNS = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'Type_encoded',
        'Temperature_difference',
        'Power'
    ]

    # 目标变量
    TARGET_BINARY = 'Is_Failure'
    TARGET_MULTI = 'Failure Type'

    # 模型保存路径
    MODEL_PATH_BINARY = 'models/rf_binary_model.pkl'
    MODEL_PATH_MULTI = 'models/rf_multi_model.pkl'
    SCALER_PATH = 'models/scaler.pkl'