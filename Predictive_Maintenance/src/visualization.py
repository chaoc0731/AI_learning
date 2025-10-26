import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from textwrap import fill


class DataVisualizer:
    def __init__(self, config):
        self.config = config
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文和英文
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_palette("husl")  # 使用更鲜明的颜色 palette

    def plot_failure_distribution(self, df):
        """绘制故障分布图 - 优化版本"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('设备故障分析总览', fontsize=16, fontweight='bold')

            # 1. 故障类型分布 - 使用条形图替代饼图
            failure_counts = df[self.config.TARGET_MULTI].value_counts()

            # 为小类别创建"其他"分组
            if len(failure_counts) > 5:
                main_categories = failure_counts.head(4)
                other_count = failure_counts[4:].sum()
                main_categories['Other'] = other_count
                plot_data = main_categories
            else:
                plot_data = failure_counts

            bars = axes[0, 0].bar(range(len(plot_data)), plot_data.values,
                                  color=plt.cm.Set3(np.linspace(0, 1, len(plot_data))))
            axes[0, 0].set_title('故障类型分布 (条形图)', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('设备数量')
            axes[0, 0].set_xticks(range(len(plot_data)))

            # 优化标签显示
            labels = [fill(str(label), 15) for label in plot_data.index]  # 自动换行
            axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')

            # 在柱子上显示数量
            for bar, count in zip(bars, plot_data.values):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + max(plot_data.values) * 0.01,
                                f'{count}', ha='center', va='bottom', fontsize=9)

            # 2. 产品类型与故障关系 - 使用堆叠百分比条形图
            if 'Type' in df.columns:
                type_failure = pd.crosstab(df['Type'], df[self.config.TARGET_MULTI], normalize='index') * 100
                # 只显示主要故障类型
                if len(type_failure.columns) > 4:
                    main_failures = type_failure.sum().nlargest(3).index
                    type_failure['Other'] = type_failure.drop(columns=main_failures).sum(axis=1)
                    type_failure = type_failure[list(main_failures) + ['Other']]

                type_failure.plot(kind='bar', stacked=True, ax=axes[0, 1],
                                  colormap='Set3', width=0.8)
                axes[0, 1].set_title('产品类型故障分布 (%)', fontsize=12, fontweight='bold')
                axes[0, 1].set_ylabel('百分比')
                axes[0, 1].set_xlabel('产品类型')
                axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[0, 1].tick_params(axis='x', rotation=45)
            else:
                axes[0, 1].text(0.5, 0.5, 'Type列不存在', ha='center', va='center', fontsize=12)
                axes[0, 1].set_title('产品类型故障分布')

            # 3. 工具磨损与故障关系 - 使用小提琴图展示分布
            if 'Tool wear [min]' in df.columns:
                # 只显示有足够样本的故障类型
                sufficient_samples = failure_counts[failure_counts >= 10].index
                filtered_data = df[df[self.config.TARGET_MULTI].isin(sufficient_samples)]

                if len(sufficient_samples) > 0:
                    sns.violinplot(data=filtered_data, x=self.config.TARGET_MULTI,
                                   y='Tool wear [min]', ax=axes[1, 0], palette='pastel')
                    axes[1, 0].set_title('工具磨损与故障类型关系', fontsize=12, fontweight='bold')
                    axes[1, 0].set_ylabel('工具磨损时间 (min)')
                    axes[1, 0].set_xlabel('故障类型')
                    axes[1, 0].tick_params(axis='x', rotation=45)
                else:
                    axes[1, 0].text(0.5, 0.5, '样本数量不足', ha='center', va='center', fontsize=12)
                    axes[1, 0].set_title('工具磨损与故障类型关系')
            else:
                axes[1, 0].text(0.5, 0.5, 'Tool wear [min]列不存在', ha='center', va='center', fontsize=12)
                axes[1, 0].set_title('工具磨损与故障类型关系')

            # 4. 故障率随时间/设备变化趋势
            if 'UDI' in df.columns:
                # 按设备序号分组计算故障率
                df_sorted = df.sort_values('UDI')
                df_sorted['Group'] = pd.cut(df_sorted['UDI'], bins=10, labels=False)
                group_failure_rate = df_sorted.groupby('Group')[self.config.TARGET_BINARY].mean() * 100

                axes[1, 1].plot(group_failure_rate.index, group_failure_rate.values,
                                marker='o', linewidth=2, markersize=6)
                axes[1, 1].set_title('故障率趋势分析', fontsize=12, fontweight='bold')
                axes[1, 1].set_ylabel('故障率 (%)')
                axes[1, 1].set_xlabel('设备分组')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                # 如果没有UDI，显示故障类型与数量的关系
                failure_stats = df.groupby(self.config.TARGET_MULTI).agg({
                    'Tool wear [min]': 'mean',
                    'Torque [Nm]': 'mean'
                }).round(2)

                if not failure_stats.empty:
                    x = range(len(failure_stats))
                    width = 0.35
                    axes[1, 1].bar(x, failure_stats['Tool wear [min]'], width,
                                   label='平均工具磨损', alpha=0.8)
                    axes[1, 1].bar([i + width for i in x], failure_stats['Torque [Nm]'], width,
                                   label='平均扭矩', alpha=0.8)
                    axes[1, 1].set_title('各故障类型的参数均值', fontsize=12, fontweight='bold')
                    axes[1, 1].set_ylabel('数值')
                    axes[1, 1].set_xlabel('故障类型')
                    axes[1, 1].set_xticks([i + width / 2 for i in x])
                    axes[1, 1].set_xticklabels([fill(str(idx), 15) for idx in failure_stats.index],
                                               rotation=45, ha='right')
                    axes[1, 1].legend()
                else:
                    axes[1, 1].text(0.5, 0.5, '无足够数据', ha='center', va='center', fontsize=12)
                    axes[1, 1].set_title('故障参数分析')

            plt.tight_layout()
            plt.subplots_adjust(top=0.94)  # 为总标题留出空间
            plt.show()

        except Exception as e:
            print(f"绘制故障分布图时出错: {e}")
            # 创建简化版本
            self.plot_simple_failure_distribution(df)

    def plot_simple_failure_distribution(self, df):
        """简化版故障分布图"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # 简单的条形图
            failure_counts = df[self.config.TARGET_MULTI].value_counts()
            bars = ax1.bar(range(len(failure_counts)), failure_counts.values,
                           color=plt.cm.tab10(np.arange(len(failure_counts))))
            ax1.set_title('故障类型分布', fontweight='bold')
            ax1.set_ylabel('设备数量')
            ax1.set_xticks(range(len(failure_counts)))
            ax1.set_xticklabels([fill(str(label), 12) for label in failure_counts.index],
                                rotation=45, ha='right')

            # 添加数值标签
            for bar, count in zip(bars, failure_counts.values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + max(failure_counts.values) * 0.01,
                         f'{count}', ha='center', va='bottom', fontsize=9)

            # 饼图（只显示主要类别）
            if len(failure_counts) > 5:
                main_categories = failure_counts.head(4)
                other_count = failure_counts[4:].sum()
                pie_data = pd.concat([main_categories, pd.Series({'Other': other_count})])
                pie_labels = [f'{label}\n({count})' for label, count in zip(pie_data.index, pie_data.values)]
            else:
                pie_data = failure_counts
                pie_labels = [f'{label}\n({count})' for label, count in
                              zip(failure_counts.index, failure_counts.values)]

            ax2.pie(pie_data.values, labels=pie_labels, autopct='%1.1f%%', startangle=90,
                    textprops={'fontsize': 9})
            ax2.set_title('故障类型占比', fontweight='bold')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"绘制简化版故障分布图时出错: {e}")

    def plot_correlation_heatmap(self, df):
        """绘制相关性热力图 - 只使用原始特征"""
        try:
            # 只使用原始数据中存在的数值列
            original_numeric_columns = [
                'Air temperature [K]',
                'Process temperature [K]',
                'Rotational speed [rpm]',
                'Torque [Nm]',
                'Tool wear [min]'
            ]

            # 过滤出数据中实际存在的列
            available_columns = [col for col in original_numeric_columns if col in df.columns]

            # 添加目标变量
            if self.config.TARGET_BINARY in df.columns:
                available_columns.append(self.config.TARGET_BINARY)

            if len(available_columns) < 2:
                print("可用的数值列不足，跳过相关性热力图")
                return

            numeric_df = df[available_columns]
            correlation_matrix = numeric_df.corr()

            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # 创建上三角掩码
            sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r',
                        center=0, fmt='.2f', square=True, mask=mask,
                        cbar_kws={"shrink": .8})
            plt.title('特征相关性热力图 (原始特征)\n', fontweight='bold')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"绘制相关性热力图时出错: {e}")

    def plot_enhanced_correlation_heatmap(self, df):
        """绘制增强版相关性热力图 - 包含工程特征"""
        try:
            # 包含所有可能的数值列
            all_numeric_columns = [
                'Air temperature [K]',
                'Process temperature [K]',
                'Rotational speed [rpm]',
                'Torque [Nm]',
                'Tool wear [min]',
                'Temperature_difference',
                'Power',
                'Temperature_ratio',
                'Power_per_torque',
                'Wear_rate',
                'Stress_index'
            ]

            # 过滤出数据中实际存在的列
            available_columns = [col for col in all_numeric_columns if col in df.columns]

            # 添加目标变量
            if self.config.TARGET_BINARY in df.columns:
                available_columns.append(self.config.TARGET_BINARY)

            if len(available_columns) < 2:
                print("可用的数值列不足，跳过增强版相关性热力图")
                return

            numeric_df = df[available_columns]
            correlation_matrix = numeric_df.corr()

            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r',
                        center=0, fmt='.2f', square=True, mask=mask,
                        cbar_kws={"shrink": .8})
            plt.title('特征相关性热力图 (包含工程特征)\n', fontweight='bold')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"绘制增强版相关性热力图时出错: {e}")

    def plot_feature_distributions(self, df):
        """绘制特征分布图"""
        try:
            # 原始数值特征
            original_features = [
                'Air temperature [K]',
                'Process temperature [K]',
                'Rotational speed [rpm]',
                'Torque [Nm]',
                'Tool wear [min]'
            ]

            available_features = [f for f in original_features if f in df.columns]

            if not available_features:
                print("没有可用的数值特征，跳过分布图")
                return

            n_features = len(available_features)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_features == 1:
                axes = [axes]
            elif n_rows > 1:
                axes = axes.flatten()

            for i, feature in enumerate(available_features):
                if i < len(axes):
                    # 直方图 + 密度曲线
                    n, bins, patches = axes[i].hist(df[feature], bins=30, alpha=0.7,
                                                    edgecolor='black', density=True)
                    axes[i].set_title(f'{feature} 分布', fontweight='bold')
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel('密度')

                    # 添加统计信息
                    mean_val = df[feature].mean()
                    std_val = df[feature].std()
                    axes[i].axvline(mean_val, color='red', linestyle='--',
                                    label=f'均值: {mean_val:.2f}')
                    axes[i].axvline(mean_val + std_val, color='orange', linestyle=':',
                                    alpha=0.7, label=f'±1标准差')
                    axes[i].axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
                    axes[i].legend(fontsize=8)

            # 隐藏多余的子图
            for i in range(len(available_features), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"绘制特征分布图时出错: {e}")

    def plot_risk_distribution(self, df, risk_scores):
        """绘制风险分布图"""
        try:
            df_with_risk = df.copy()
            df_with_risk['Failure_Risk_Score'] = risk_scores * 100

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # 风险分布直方图
            n, bins, patches = axes[0].hist(df_with_risk['Failure_Risk_Score'], bins=50,
                                            alpha=0.7, edgecolor='black', density=True)
            axes[0].axvline(70, color='red', linestyle='--', linewidth=2,
                            label='高风险阈值 (70)')
            axes[0].axvline(30, color='green', linestyle='--', linewidth=2,
                            label='低风险阈值 (30)')
            axes[0].set_xlabel('故障风险评分')
            axes[0].set_ylabel('密度')
            axes[0].set_title('设备故障风险分布', fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 按产品类型的平均风险
            if 'Type' in df_with_risk.columns:
                risk_by_type = df_with_risk.groupby('Type')['Failure_Risk_Score'].mean().sort_values()
                bars = axes[1].bar(range(len(risk_by_type)), risk_by_type.values,
                                   color=plt.cm.viridis(np.linspace(0, 1, len(risk_by_type))))
                axes[1].set_title('各产品类型的平均故障风险', fontweight='bold')
                axes[1].set_ylabel('平均风险评分')
                axes[1].set_xlabel('产品类型')
                axes[1].set_xticks(range(len(risk_by_type)))
                axes[1].set_xticklabels(risk_by_type.index, rotation=45, ha='right')

                # 添加数值标签
                for bar, risk in zip(bars, risk_by_type.values):
                    height = bar.get_height()
                    axes[1].text(bar.get_x() + bar.get_width() / 2., height + 1,
                                 f'{risk:.1f}', ha='center', va='bottom', fontsize=9)
            else:
                # 如果没有Type列，按故障状态分组
                if self.config.TARGET_BINARY in df_with_risk.columns:
                    risk_by_failure = df_with_risk.groupby(self.config.TARGET_BINARY)['Failure_Risk_Score'].mean()
                    colors = ['green', 'red']
                    bars = axes[1].bar(['正常', '故障'], risk_by_failure.values, color=colors)
                    axes[1].set_title('故障与非故障设备的平均风险', fontweight='bold')
                    axes[1].set_ylabel('平均风险评分')

                    # 添加数值标签
                    for bar, risk in zip(bars, risk_by_failure.values):
                        height = bar.get_height()
                        axes[1].text(bar.get_x() + bar.get_width() / 2., height + 1,
                                     f'{risk:.1f}', ha='center', va='bottom', fontweight='bold')
                else:
                    axes[1].text(0.5, 0.5, '无分组信息', ha='center', va='center', fontsize=12)
                    axes[1].set_title('风险分布')

            plt.tight_layout()
            plt.show()

            return df_with_risk

        except Exception as e:
            print(f"绘制风险分布图时出错: {e}")
            return df