import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()

    def create_features(self, df):
        """创建新特征"""
        print("开始特征工程...")
        print(f"输入数据形状: {df.shape}")
        print(f"输入数据列: {list(df.columns)}")

        try:
            # 温度相关特征
            df['Temperature_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
            df['Temperature_ratio'] = df['Process temperature [K]'] / df['Air temperature [K]']

            # 功率相关特征
            df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]'] / 9549
            df['Power_per_torque'] = df['Power'] / (df['Torque [Nm]'] + 1e-6)

            # 工具磨损相关特征
            df['Wear_rate'] = df['Tool wear [min]'] / (df['Rotational speed [rpm]'] + 1e-6)

            # 综合特征
            df['Stress_index'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] / (df['Tool wear [min]'] + 1)

            print(f"特征工程完成，新增特征后的数据形状: {df.shape}")
            print(f"新增特征后的数据列: {list(df.columns)}")

            return df

        except Exception as e:
            print(f"特征工程时出错: {e}")
            return df

    def prepare_features(self, df, fit_scaler=True):
        """准备模型特征"""
        try:
            # 首先创建新特征
            features_df = self.create_features(df)

            # 动态获取可用的特征列
            available_features = []
            for feature in self.config.FEATURE_COLUMNS:
                if feature in features_df.columns:
                    available_features.append(feature)
                else:
                    print(f"警告: 特征 {feature} 在数据中不存在，已跳过")

            print(f"使用的特征列表: {available_features}")

            if not available_features:
                print("❌ 错误: 没有可用的特征")
                return None, None, None

            # 选择最终特征
            X = features_df[available_features]

            # 检查是否有NaN值
            if X.isnull().any().any():
                print("警告: 特征中存在NaN值，进行填充")
                X = X.fillna(X.mean())

            # 标准化特征
            if fit_scaler:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)

            print(f"特征准备完成 - 标准化后形状: {X_scaled.shape}")

            return X_scaled, features_df, available_features

        except Exception as e:
            print(f"准备特征时出错: {e}")
            return None, None, None

    def get_feature_names(self, df=None):
        """获取可用的特征名称"""
        if df is not None:
            # 返回数据中实际存在的特征
            available_features = [f for f in self.config.FEATURE_COLUMNS if f in df.columns]
            return available_features
        return self.config.FEATURE_COLUMNS

    def save_scaler(self, path):
        """保存标准化器"""
        try:
            joblib.dump(self.scaler, path)
            print(f"标准化器已保存至: {path}")
        except Exception as e:
            print(f"保存标准化器时出错: {e}")

    def load_scaler(self, path):
        """加载标准化器"""
        try:
            self.scaler = joblib.load(path)
            print(f"标准化器已从 {path} 加载")
        except Exception as e:
            print(f"加载标准化器时出错: {e}")