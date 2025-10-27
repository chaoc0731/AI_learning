import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from commerce_analysis.config.setting import Config


logger = logging.getLogger(__name__)


class RFMAnalyzer:
    """RFM分析器 - 修复分箱问题"""

    def __init__(self):
        self.config = Config

    def calculate_rfm(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """计算RFM指标"""
        logger.info("开始计算RFM指标...")

        # 获取分析时间点
        analysis_date = events_df['timestamp'].max()

        # 计算RFM指标
        rfm = events_df.groupby('visitorid').agg({
            'timestamp': lambda x: (analysis_date - x.max()).days,  # Recency
            'event': 'count',  # Frequency
            'event_weight': 'sum'  # Monetary
        }).reset_index()

        rfm.columns = ['visitorid', 'recency', 'frequency', 'monetary']

        logger.info(f"RFM指标计算完成，用户数: {len(rfm)}")

        # 记录RFM指标的统计信息
        logger.info(f"Recency范围: {rfm['recency'].min()} - {rfm['recency'].max()}")
        logger.info(f"Frequency范围: {rfm['frequency'].min()} - {rfm['frequency'].max()}")
        logger.info(f"Monetary范围: {rfm['monetary'].min()} - {rfm['monetary'].max()}")

        # 检查频率分布
        freq_stats = rfm['frequency'].value_counts().head(10)
        logger.info(f"Frequency前10分布:\n{freq_stats}")

        return rfm

    def calculate_rfm_scores(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """计算RFM分数 - 修复分箱问题"""
        df = rfm_df.copy()

        # 检查数据分布
        logger.info("检查RFM指标分布...")
        for col in ['recency', 'frequency', 'monetary']:
            unique_values = df[col].nunique()
            logger.info(f"{col}: {unique_values} 个唯一值")

        # 使用安全的分箱方法
        df['R_Score'] = self._safe_qcut(df['recency'], 4, [4, 3, 2, 1])
        df['F_Score'] = self._safe_qcut(df['frequency'], 4, [1, 2, 3, 4])
        df['M_Score'] = self._safe_qcut(df['monetary'], 4, [1, 2, 3, 4])

        # 计算RFM总得分
        df['RFM_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']

        logger.info("RFM分数计算完成")
        return df

    def _safe_qcut(self, series, q, labels):
        """
        安全的分位数切分方法
        处理重复边界值的情况
        """
        try:
            # 首先尝试常规分位数切分
            return pd.qcut(series, q, labels=labels, duplicates='drop')
        except ValueError as e:
            logger.warning(f"常规分位数切分失败: {e}，使用替代方法")

            # 方法1: 使用等宽分箱
            try:
                return pd.cut(series, bins=q, labels=labels, include_lowest=True)
            except:
                # 方法2: 手动分箱
                return self._manual_binning(series, q, labels)

    def _manual_binning(self, series, q, labels):
        """
        手动分箱方法
        当自动分箱失败时使用
        """
        # 计算实际的分位数
        quantiles = []
        for i in range(q + 1):
            try:
                quantile_val = series.quantile(i / q)
                quantiles.append(quantile_val)
            except:
                quantiles.append(series.min() + (series.max() - series.min()) * i / q)

        # 移除重复的分界点
        unique_quantiles = []
        for q_val in quantiles:
            if not unique_quantiles or q_val > unique_quantiles[-1]:
                unique_quantiles.append(q_val)

        # 如果唯一分界点不够，使用等间距分箱
        if len(unique_quantiles) < q + 1:
            logger.info("使用等间距分箱")
            bins = np.linspace(series.min(), series.max(), q + 1)
            return pd.cut(series, bins=bins, labels=labels, include_lowest=True)

        # 使用唯一分界点进行分箱
        return pd.cut(series, bins=unique_quantiles, labels=labels, include_lowest=True)

    def calculate_rfm_scores_robust(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        更健壮的RFM分数计算方法
        处理各种边缘情况
        """
        df = rfm_df.copy()

        # 方法1: 使用排名而不是分位数
        logger.info("使用排名方法计算RFM分数...")

        # 对于Recency，值越小越好（最近活跃）
        df['R_Score'] = pd.qcut(df['recency'].rank(method='first'), 4, labels=[4, 3, 2, 1], duplicates='drop')

        # 对于Frequency和Monetary，值越大越好
        df['F_Score'] = pd.qcut(df['frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')
        df['M_Score'] = pd.qcut(df['monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')

        # 处理可能的NaN值
        for col in ['R_Score', 'F_Score', 'M_Score']:
            df[col] = df[col].fillna(2)  # 用中位数填充

        # 计算RFM总得分
        df['RFM_Score'] = df['R_Score'].astype(int) + df['F_Score'].astype(int) + df['M_Score'].astype(int)

        return df

    def segment_users(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """用户分群"""
        df = rfm_df.copy()

        def get_segment(row):
            score = row['RFM_Score']
            if score >= self.config.RFM_SEGMENT_CONFIG['high_value']:
                return '高价值用户'
            elif score >= self.config.RFM_SEGMENT_CONFIG['potential']:
                return '潜力用户'
            elif score >= self.config.RFM_SEGMENT_CONFIG['general']:
                return '一般用户'
            else:
                return '流失风险用户'

        df['user_segment'] = df.apply(get_segment, axis=1)

        # 统计分群结果
        segment_stats = df['user_segment'].value_counts()
        logger.info(f"用户分群完成: {dict(segment_stats)}")

        return df

    def perform_clustering(self, rfm_df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
        """使用K-means进行聚类分析"""
        df = rfm_df.copy()

        # 选择特征并标准化
        features = ['recency', 'frequency', 'monetary']
        X = df[features]

        # 检查数据有效性
        if X.isna().any().any():
            logger.warning("存在空值，进行填充")
            X = X.fillna(X.median())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)

        logger.info(f"K-means聚类完成，聚类数: {n_clusters}")
        return df

    def get_segment_analysis(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """获取分群分析结果"""
        segment_analysis = (rfm_df.groupby('user_segment')
                            .agg({
            'visitorid': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'RFM_Score': 'mean'
        })
                            .round(2)
                            .rename(columns={'visitorid': '用户数'}))

        segment_analysis['占比'] = (segment_analysis['用户数'] / segment_analysis['用户数'].sum() * 100).round(2)

        return segment_analysis

    def analyze_rfm_distribution(self, rfm_df: pd.DataFrame):
        """分析RFM分布情况"""
        logger.info("RFM分布分析:")

        for col in ['recency', 'frequency', 'monetary']:
            stats = {
                'min': rfm_df[col].min(),
                'max': rfm_df[col].max(),
                'mean': rfm_df[col].mean(),
                'median': rfm_df[col].median(),
                'std': rfm_df[col].std(),
                'unique': rfm_df[col].nunique()
            }
            logger.info(f"{col}: {stats}")

            # 检查值的分布
            value_counts = rfm_df[col].value_counts().head(5)
            logger.info(f"{col} 前5值分布: {dict(value_counts)}")