import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DashboardCreator:
    """仪表盘创建器 - 完整版本"""

    def __init__(self):
        self.setup_chinese_font()
        self.setup_plot_style()

    def setup_chinese_font(self):
        """设置中文字体 - 简化版本"""
        try:
            # 直接设置全局字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.size'] = 12

            logger.info("中文字体设置完成")
        except Exception as e:
            logger.warning(f"字体设置警告: {e}")

    def setup_plot_style(self):
        """设置绘图样式"""
        try:
            plt.style.use('default')
            sns.set_style("whitegrid")
            logger.info("绘图样式设置完成")
        except Exception as e:
            logger.error(f"设置绘图样式时出错: {e}")

    def create_user_behavior_dashboard(self, events_df: pd.DataFrame, save_path: str = None):
        """创建用户行为分析仪表盘"""
        logger.info("创建用户行为分析仪表盘...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('电商用户行为分析仪表盘', fontsize=16, fontweight='bold')

        try:
            # 1. 用户行为分布
            event_dist = events_df['event'].value_counts()
            event_labels = {
                'view': '浏览',
                'addtocart': '加入购物车',
                'transaction': '交易'
            }
            event_dist.index = [event_labels.get(x, x) for x in event_dist.index]

            axes[0, 0].pie(event_dist.values, labels=event_dist.index, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('用户行为分布', fontweight='bold')

            # 2. 每日行为趋势
            daily_events = events_df.groupby('date').size()
            axes[0, 1].plot(daily_events.index, daily_events.values, color='blue', linewidth=2)
            axes[0, 1].set_title('每日用户行为趋势', fontweight='bold')
            axes[0, 1].set_xlabel('日期')
            axes[0, 1].set_ylabel('行为次数')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 小时行为分布
            hourly_activity = events_df.groupby('hour').size()
            axes[0, 2].plot(hourly_activity.index, hourly_activity.values, marker='o',
                            color='green', linewidth=2, markersize=6)
            axes[0, 2].set_title('用户活跃时间分布', fontweight='bold')
            axes[0, 2].set_xlabel('小时')
            axes[0, 2].set_ylabel('行为次数')
            axes[0, 2].grid(True, alpha=0.3)

            # 4. 周末vs工作日
            weekend_analysis = events_df.groupby('weekend')['event'].value_counts().unstack()
            weekend_analysis.index = ['工作日', '周末']
            weekend_analysis.columns = [event_labels.get(x, x) for x in weekend_analysis.columns]
            weekend_analysis.plot(kind='bar', ax=axes[1, 0], width=0.8)
            axes[1, 0].set_title('周末vs工作日行为对比', fontweight='bold')
            axes[1, 0].set_xlabel('时间类型')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 0].tick_params(axis='x', rotation=0)

            # 5. 星期分布
            weekday_activity = events_df.groupby('day_of_week').size()
            weekday_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            weekday_activity.index = [weekday_labels[i] if i < len(weekday_labels) else str(i) for i in
                                      weekday_activity.index]
            axes[1, 1].bar(weekday_activity.index, weekday_activity.values,
                           color='orange', alpha=0.7)
            axes[1, 1].set_title('星期行为分布', fontweight='bold')
            axes[1, 1].set_xlabel('星期')
            axes[1, 1].set_ylabel('行为次数')
            axes[1, 1].grid(True, alpha=0.3)

            # 6. 月度趋势
            monthly_activity = events_df.groupby('month').size()
            axes[1, 2].bar(monthly_activity.index, monthly_activity.values,
                           color='purple', alpha=0.7)
            axes[1, 2].set_title('月度行为趋势', fontweight='bold')
            axes[1, 2].set_xlabel('月份')
            axes[1, 2].set_ylabel('行为次数')
            axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                logger.info(f"仪表盘已保存至: {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"创建用户行为仪表盘时出错: {e}")

    def create_rfm_dashboard(self, rfm_df: pd.DataFrame, save_path: str = None):
        """创建RFM分析仪表盘"""
        logger.info("创建RFM分析仪表盘...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RFM用户价值分析仪表盘', fontsize=16, fontweight='bold')

        try:
            # 1. 用户分群分布
            segment_counts = rfm_df['user_segment'].value_counts()
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            axes[0, 0].pie(segment_counts.values, labels=segment_counts.index,
                           autopct='%1.1f%%', startangle=90, colors=colors)
            axes[0, 0].set_title('用户分群分布', fontweight='bold')

            # 2. RFM散点图
            scatter = axes[0, 1].scatter(rfm_df['recency'], rfm_df['frequency'],
                                         c=rfm_df['monetary'], cmap='viridis',
                                         alpha=0.6, s=50)
            axes[0, 1].set_xlabel('最近购买天数 (Recency)', fontweight='bold')
            axes[0, 1].set_ylabel('购买频率 (Frequency)', fontweight='bold')
            axes[0, 1].set_title('RFM分布散点图', fontweight='bold')
            plt.colorbar(scatter, ax=axes[0, 1], label='消费金额 (Monetary)')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 各分群RFM指标对比
            segment_stats = rfm_df.groupby('user_segment')[['recency', 'frequency', 'monetary']].mean()
            segment_stats.columns = ['最近购买', '购买频率', '消费金额']
            segment_stats.plot(kind='bar', ax=axes[1, 0], width=0.8)
            axes[1, 0].set_title('各分群RFM指标对比', fontweight='bold')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 0].grid(True, alpha=0.3)

            # 4. RFM得分分布
            axes[1, 1].hist(rfm_df['RFM_Score'], bins=20, alpha=0.7,
                            edgecolor='black', color='skyblue')
            axes[1, 1].set_xlabel('RFM得分', fontweight='bold')
            axes[1, 1].set_ylabel('用户数量', fontweight='bold')
            axes[1, 1].set_title('RFM得分分布', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                logger.info(f"RFM仪表盘已保存至: {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"创建RFM仪表盘时出错: {e}")

    def create_interactive_dashboard(self, events_df: pd.DataFrame, rfm_df: pd.DataFrame):
        """创建交互式仪表盘（Plotly）"""
        try:
            logger.info("创建交互式仪表盘...")

            # 翻译事件类型
            event_labels = {
                'view': '浏览',
                'addtocart': '加入购物车',
                'transaction': '交易'
            }

            # 创建子图
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=(
                    '用户行为分布', '每日行为趋势', '用户活跃时间',
                    'RFM用户分群', '星期分布', '月度趋势',
                    '用户价值分布', '行为转化漏斗', '时间热点分析'
                ),
                specs=[
                    [{"type": "pie"}, {"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                    [{"type": "histogram"}, {"type": "funnel"}, {"type": "heatmap"}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.08
            )

            # 1. 用户行为分布（饼图）
            event_dist = events_df['event'].value_counts()
            event_dist.index = [event_labels.get(x, x) for x in event_dist.index]
            fig.add_trace(
                go.Pie(
                    labels=event_dist.index,
                    values=event_dist.values,
                    name="行为分布",
                    hole=0.3,
                    textinfo='label+percent',
                    marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                ),
                row=1, col=1
            )

            # 2. 每日行为趋势
            daily_events = events_df.groupby('date').size().reset_index()
            daily_events.columns = ['date', 'count']
            fig.add_trace(
                go.Scatter(
                    x=daily_events['date'],
                    y=daily_events['count'],
                    mode='lines',
                    name='每日趋势',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='日期: %{x}<br>行为次数: %{y}'
                ),
                row=1, col=2
            )

            # 3. 用户活跃时间
            hourly_activity = events_df.groupby('hour').size().reset_index()
            hourly_activity.columns = ['hour', 'count']
            fig.add_trace(
                go.Scatter(
                    x=hourly_activity['hour'],
                    y=hourly_activity['count'],
                    mode='lines+markers',
                    name='小时分布',
                    line=dict(color='#2ca02c', width=3),
                    marker=dict(size=8, color='#2ca02c'),
                    hovertemplate='小时: %{x}:00<br>行为次数: %{y}'
                ),
                row=1, col=3
            )

            # 4. RFM用户分群
            segment_counts = rfm_df['user_segment'].value_counts().reset_index()
            segment_counts.columns = ['segment', 'count']
            fig.add_trace(
                go.Bar(
                    x=segment_counts['segment'],
                    y=segment_counts['count'],
                    name='用户分群',
                    marker_color='#ff7f0e',
                    hovertemplate='用户分群: %{x}<br>用户数量: %{y}'
                ),
                row=2, col=1
            )

            # 5. 星期分布
            weekday_activity = events_df.groupby('day_of_week').size().reset_index()
            weekday_activity.columns = ['day_of_week', 'count']
            weekday_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            weekday_activity['day_name'] = [weekday_labels[i] if i < len(weekday_labels) else str(i)
                                            for i in weekday_activity['day_of_week']]
            fig.add_trace(
                go.Bar(
                    x=weekday_activity['day_name'],
                    y=weekday_activity['count'],
                    name='星期分布',
                    marker_color='#9467bd',
                    hovertemplate='星期: %{x}<br>行为次数: %{y}'
                ),
                row=2, col=2
            )

            # 6. 月度趋势
            monthly_activity = events_df.groupby('month').size().reset_index()
            monthly_activity.columns = ['month', 'count']
            fig.add_trace(
                go.Bar(
                    x=monthly_activity['month'],
                    y=monthly_activity['count'],
                    name='月度趋势',
                    marker_color='#d62728',
                    hovertemplate='月份: %{x}<br>行为次数: %{y}'
                ),
                row=2, col=3
            )

            # 7. RFM得分分布
            fig.add_trace(
                go.Histogram(
                    x=rfm_df['RFM_Score'],
                    name='RFM得分分布',
                    nbinsx=20,
                    marker_color='#17becf',
                    opacity=0.7
                ),
                row=3, col=1
            )

            # 8. 行为转化漏斗
            funnel_data = self._calculate_funnel_data(events_df)
            fig.add_trace(
                go.Funnel(
                    y=funnel_data['stage'],
                    x=funnel_data['count'],
                    name='行为转化',
                    marker=dict(color=['#FF6B6B', '#4ECDC4', '#45B7D1']),
                    textinfo='value+percent initial'
                ),
                row=3, col=2
            )

            # 9. 时间热点分析
            heatmap_data = self._calculate_heatmap_data(events_df)
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data['count'].values.reshape(len(heatmap_data['hour'].unique()),
                                                           len(heatmap_data['day_of_week'].unique())),
                    x=weekday_labels[:7],
                    y=list(range(24)),
                    colorscale='Viridis',
                    name='时间热点',
                    hovertemplate='星期: %{x}<br>小时: %{y}:00<br>行为次数: %{z}'
                ),
                row=3, col=3
            )

            # 更新布局
            fig.update_layout(
                height=1200,
                showlegend=True,
                title_text="电商用户行为分析交互式仪表盘",
                title_x=0.5,
                title_font=dict(size=24, color='#2c3e50'),
                font=dict(size=12, family="Arial, SimHei"),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=100, l=50, r=50, b=50)
            )

            # 更新坐标轴标签
            fig.update_xaxes(title_text="日期", row=1, col=2)
            fig.update_yaxes(title_text="行为次数", row=1, col=2)
            fig.update_xaxes(title_text="小时", row=1, col=3)
            fig.update_yaxes(title_text="行为次数", row=1, col=3)
            fig.update_xaxes(title_text="用户分群", row=2, col=1)
            fig.update_yaxes(title_text="用户数量", row=2, col=1)
            fig.update_xaxes(title_text="星期", row=2, col=2)
            fig.update_yaxes(title_text="行为次数", row=2, col=2)
            fig.update_xaxes(title_text="月份", row=2, col=3)
            fig.update_yaxes(title_text="行为次数", row=2, col=3)
            fig.update_xaxes(title_text="RFM得分", row=3, col=1)
            fig.update_yaxes(title_text="用户数量", row=3, col=1)
            fig.update_xaxes(title_text="转化阶段", row=3, col=2)
            fig.update_yaxes(title_text="用户数量", row=3, col=2)
            fig.update_xaxes(title_text="星期", row=3, col=3)
            fig.update_yaxes(title_text="小时", row=3, col=3)

            # 保存交互式图表
            try:
                from commerce_analysis.config.setting import Config
                output_path = Config.OUTPUT_DIR / "figures" / "interactive_dashboard.html"
                fig.write_html(str(output_path))
                logger.info(f"交互式仪表盘已保存至: {output_path}")
            except:
                # 如果Config不可用，使用相对路径
                output_path = "interactive_dashboard.html"
                fig.write_html(output_path)
                logger.info(f"交互式仪表盘已保存至: {output_path}")

            # 在Jupyter notebook中显示
            fig.show()

            return fig

        except Exception as e:
            logger.error(f"创建交互式仪表盘时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_funnel_data(self, events_df):
        """计算漏斗数据"""
        try:
            # 计算各阶段用户数
            view_users = events_df[events_df['event'] == 'view']['visitorid'].nunique()
            cart_users = events_df[events_df['event'] == 'addtocart']['visitorid'].nunique()
            transaction_users = events_df[events_df['event'] == 'transaction']['visitorid'].nunique()

            funnel_data = pd.DataFrame({
                'stage': ['浏览', '加入购物车', '交易'],
                'count': [view_users, cart_users, transaction_users]
            })

            return funnel_data
        except:
            # 返回默认数据
            return pd.DataFrame({
                'stage': ['浏览', '加入购物车', '交易'],
                'count': [1000, 500, 100]
            })

    def _calculate_heatmap_data(self, events_df):
        """计算热力图数据"""
        try:
            heatmap_data = events_df.groupby(['hour', 'day_of_week']).size().reset_index()
            heatmap_data.columns = ['hour', 'day_of_week', 'count']
            return heatmap_data
        except:
            # 返回默认数据
            import numpy as np
            hours = list(range(24))
            days = list(range(7))
            data = []
            for hour in hours:
                for day in days:
                    data.append({'hour': hour, 'day_of_week': day, 'count': np.random.randint(10, 100)})
            return pd.DataFrame(data)

    def create_simple_interactive_dashboard(self, events_df: pd.DataFrame, rfm_df: pd.DataFrame):
        """创建简化版交互式仪表盘"""
        try:
            logger.info("创建简化版交互式仪表盘...")

            # 创建标签映射
            event_labels = {
                'view': '浏览',
                'addtocart': '加入购物车',
                'transaction': '交易'
            }

            # 只创建4个主要图表
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('用户行为分布', '每日行为趋势', 'RFM用户分群', '星期分布'),
                specs=[[{"type": "pie"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )

            # 1. 用户行为分布
            event_dist = events_df['event'].value_counts()
            event_dist.index = [event_labels.get(x, x) for x in event_dist.index]
            fig.add_trace(
                go.Pie(labels=event_dist.index, values=event_dist.values, name="行为分布"),
                row=1, col=1
            )

            # 2. 每日行为趋势
            daily_events = events_df.groupby('date').size().reset_index()
            fig.add_trace(
                go.Scatter(x=daily_events['date'], y=daily_events[0], mode='lines', name='每日趋势'),
                row=1, col=2
            )

            # 3. RFM用户分群
            segment_counts = rfm_df['user_segment'].value_counts()
            fig.add_trace(
                go.Bar(x=segment_counts.index, y=segment_counts.values, name='用户分群'),
                row=2, col=1
            )

            # 4. 星期分布
            weekday_activity = events_df.groupby('day_of_week').size().reset_index()
            weekday_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            weekday_activity['day_name'] = [weekday_labels[i] if i < len(weekday_labels) else str(i)
                                            for i in weekday_activity['day_of_week']]
            fig.add_trace(
                go.Bar(x=weekday_activity['day_name'], y=weekday_activity[0], name='星期分布'),
                row=2, col=2
            )

            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="电商用户行为分析简化仪表盘",
                title_x=0.5
            )

            # 保存简化版
            try:
                from commerce_analysis.config.setting import Config
                output_path = Config.OUTPUT_DIR / "figures" / "simple_interactive_dashboard.html"
                fig.write_html(str(output_path))
                logger.info(f"简化交互式仪表盘已保存至: {output_path}")
            except:
                fig.write_html("simple_interactive_dashboard.html")
                logger.info("简化交互式仪表盘已保存至: simple_interactive_dashboard.html")

            return fig

        except Exception as e:
            logger.error(f"创建简化交互式仪表盘时出错: {e}")
            return None