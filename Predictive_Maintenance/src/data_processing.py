import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_data(self, file_path=None):
        """加载原始数据"""
        if file_path is None:
            file_path = self.config.RAW_DATA_PATH

        print(f"正在加载数据: {file_path}")
        df = pd.read_csv(file_path)
        print(f"数据加载完成，形状: {df.shape}")
        return df

    def clean_data(self, df):
        """数据清洗"""
        print("开始数据清洗...")

        # 检查缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"发现缺失值: \n{missing_values[missing_values > 0]}")
            # 这里可以根据业务逻辑处理缺失值
            df = df.dropna()

        # 检查重复值
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"发现 {duplicates} 个重复值，已删除")
            df = df.drop_duplicates()

        print(f"清洗后数据形状: {df.shape}")
        return df

    def encode_categorical(self, df):
        """编码分类变量"""
        if 'Type' in df.columns:
            df['Type_encoded'] = self.label_encoder.fit_transform(df['Type'])
            print(
                f"产品类型编码完成: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        return df

    def create_target_variables(self, df):
        """创建目标变量"""
        df['Is_Failure'] = (df['Failure Type'] != 'No Failure').astype(int)
        print(f"目标变量创建完成 - 故障率: {df['Is_Failure'].mean():.3f}")
        return df

    def save_processed_data(self, df):
        """保存处理后的数据"""
        os.makedirs(os.path.dirname(self.config.PROCESSED_DATA_PATH), exist_ok=True)
        df.to_csv(self.config.PROCESSED_DATA_PATH, index=False)
        print(f"处理后的数据已保存至: {self.config.PROCESSED_DATA_PATH}")

    def process_data(self, file_path=None):
        """完整的数据处理流程"""
        df = self.load_data(file_path)
        df = self.clean_data(df)
        df = self.encode_categorical(df)
        df = self.create_target_variables(df)
        self.save_processed_data(df)
        return df