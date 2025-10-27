import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from commerce_analysis.config.setting import Config


logger = logging.getLogger(__name__)


class BehaviorPredictor:
    """用户行为预测器"""

    def __init__(self):
        self.config = Config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None

    def prepare_features(self, events_df: pd.DataFrame, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        logger.info("开始准备特征数据...")

        # 用户活跃度特征
        user_activity = events_df.groupby('visitorid').agg({
            'event': 'count',
            'timestamp': ['min', 'max'],
            'event_weight': 'sum'
        }).round(2)

        user_activity.columns = ['total_events', 'first_activity', 'last_activity', 'engagement_score']
        user_activity['activity_days'] = (user_activity['last_activity'] - user_activity['first_activity']).dt.days

        # 合并RFM特征
        features_df = user_activity.merge(
            rfm_df[['visitorid', 'recency', 'frequency', 'monetary', 'user_segment']],
            left_index=True,
            right_on='visitorid'
        )

        # 创建目标变量（是否完成过交易）
        transaction_users = events_df[events_df['event'] == 'transaction']['visitorid'].unique()
        features_df['has_transaction'] = features_df['visitorid'].isin(transaction_users).astype(int)

        # 处理分类变量
        features_df = pd.get_dummies(features_df, columns=['user_segment'], prefix='segment')

        logger.info(f"特征准备完成，形状: {features_df.shape}")
        return features_df

    def train_model(self, features_df: pd.DataFrame, target_column: str = 'has_transaction'):
        """训练预测模型"""
        logger.info("开始训练预测模型...")

        # 选择特征列
        feature_columns = [col for col in features_df.columns
                           if col not in ['visitorid', 'first_activity', 'last_activity', target_column]]

        X = features_df[feature_columns].fillna(0)
        y = features_df[target_column]

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )

        # 训练随机森林模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=10
        )

        self.model.fit(X_train, y_train)

        # 模型评估
        y_pred = self.model.predict(X_test)

        logger.info("模型训练完成")

        # 特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return X_test, y_test, y_pred

    def evaluate_model(self, X_test, y_test, y_pred):
        """评估模型性能"""
        logger.info("开始模型评估...")

        # 分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # 转换为DataFrame便于展示
        report_df = pd.DataFrame(report).transpose()

        logger.info("模型评估完成")
        return report_df, cm

    def save_model(self, filename: str = 'behavior_predictor.joblib'):
        """保存模型"""
        if self.model is None:
            logger.error("没有训练好的模型可以保存")
            return

        model_path = self.config.MODEL_DIR / filename
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }, model_path)

        logger.info(f"模型已保存至: {model_path}")

    def load_model(self, filename: str = 'behavior_predictor.joblib'):
        """加载模型"""
        model_path = self.config.MODEL_DIR / filename

        try:
            loaded_data = joblib.load(model_path)
            self.model = loaded_data['model']
            self.scaler = loaded_data['scaler']
            self.feature_importance = loaded_data['feature_importance']
            logger.info(f"模型已从 {model_path} 加载")
        except FileNotFoundError:
            logger.error(f"未找到模型文件: {model_path}")
            raise