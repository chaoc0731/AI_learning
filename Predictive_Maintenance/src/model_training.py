from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.binary_model = None
        self.multi_model = None
        self.available_features = []

    def prepare_data(self, df, feature_engineer):
        """准备训练数据"""
        try:
            X, features_df, available_features = feature_engineer.prepare_features(df)
            self.available_features = available_features  # 保存实际可用的特征

            y_binary = features_df[self.config.TARGET_BINARY]
            y_multi = features_df[self.config.TARGET_MULTI]

            print(f"训练数据准备完成 - 特征维度: {X.shape}, 二分类目标: {y_binary.shape}, 多分类目标: {y_multi.shape}")

            # 返回包含工程特征的DataFrame用于可视化
            return X, y_binary, y_multi, features_df
        except Exception as e:
            print(f"准备数据时出错: {e}")
            return None, None, None, None

    def train_binary_model(self, X, y, optimize=False):
        """训练二分类模型（是否故障）"""
        if X is None or y is None:
            print("错误: 训练数据为空")
            return None, None, None

        print("开始训练二分类模型...")

        try:
            # 划分训练测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE, stratify=y
            )

            print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
            print(f"训练集故障比例: {y_train.mean():.3f}, 测试集故障比例: {y_test.mean():.3f}")

            if optimize:
                # 网格搜索优化参数
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }

                rf = RandomForestClassifier(random_state=self.config.RANDOM_STATE)
                grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
                grid_search.fit(X_train, y_train)

                self.binary_model = grid_search.best_estimator_
                print(f"最佳参数: {grid_search.best_params_}")
            else:
                # 使用默认参数
                self.binary_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.RANDOM_STATE,
                    class_weight='balanced'  # 处理类别不平衡
                )
                self.binary_model.fit(X_train, y_train)

            # 评估模型
            y_pred = self.binary_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"二分类模型训练完成，测试集准确率: {accuracy:.4f}")

            return X_test, y_test, y_pred

        except Exception as e:
            print(f"训练二分类模型时出错: {e}")
            return None, None, None

    def train_multi_model(self, X, y, optimize=False):
        """训练多分类模型（故障类型）"""
        if X is None or y is None:
            print("错误: 训练数据为空")
            return None, None, None

        print("开始训练多分类模型...")

        try:
            # 划分训练测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE, stratify=y
            )

            print(f"多分类训练集: {X_train.shape}, 测试集: {X_test.shape}")
            print(f"多分类标签分布:\n{y_train.value_counts()}")

            if optimize:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                }

                rf = RandomForestClassifier(random_state=self.config.RANDOM_STATE, class_weight='balanced')
                grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
                grid_search.fit(X_train, y_train)

                self.multi_model = grid_search.best_estimator_
                print(f"最佳参数: {grid_search.best_params_}")
            else:
                self.multi_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.RANDOM_STATE,
                    class_weight='balanced'
                )
                self.multi_model.fit(X_train, y_train)

            # 评估模型
            y_pred = self.multi_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"多分类模型训练完成，测试集准确率: {accuracy:.4f}")

            return X_test, y_test, y_pred

        except Exception as e:
            print(f"训练多分类模型时出错: {e}")
            return None, None, None

    def get_available_features(self):
        """获取实际可用的特征列表"""
        return self.available_features

    def save_models(self):
        """保存训练好的模型"""
        import os
        os.makedirs('models', exist_ok=True)

        if self.binary_model:
            joblib.dump(self.binary_model, self.config.MODEL_PATH_BINARY)
            print(f"二分类模型已保存至: {self.config.MODEL_PATH_BINARY}")
        else:
            print("警告: 二分类模型为空，跳过保存")

        if self.multi_model:
            joblib.dump(self.multi_model, self.config.MODEL_PATH_MULTI)
            print(f"多分类模型已保存至: {self.config.MODEL_PATH_MULTI}")
        else:
            print("警告: 多分类模型为空，跳过保存")

        # 保存特征列表
        if self.available_features:
            feature_info = {
                'available_features': self.available_features
            }
            joblib.dump(feature_info, 'models/feature_info.pkl')
            print("特征信息已保存")
        else:
            print("警告: 特征列表为空，跳过保存")

    def load_models(self):
        """加载已训练的模型"""
        try:
            self.binary_model = joblib.load(self.config.MODEL_PATH_BINARY)
            self.multi_model = joblib.load(self.config.MODEL_PATH_MULTI)

            # 加载特征信息
            feature_info = joblib.load('models/feature_info.pkl')
            self.available_features = feature_info['available_features']

            print("模型和特征信息加载完成")
        except Exception as e:
            print(f"加载模型时出错: {e}")