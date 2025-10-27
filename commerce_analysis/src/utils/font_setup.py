"""
全局字体设置模块
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import logging
import os
import sys

logger = logging.getLogger(__name__)


class FontSetup:
    """字体设置类"""

    def __init__(self):
        self.setup_complete = False

    def setup_chinese_font(self):
        """设置中文字体 - 深度修复版本"""
        if self.setup_complete:
            return True

        try:
            logger.info("开始设置中文字体...")

            # 方法1: 直接设置matplotlib配置
            mpl.rcParams['font.sans-serif'] = [
                'SimHei',  # 黑体
                'Microsoft YaHei',  # 微软雅黑
                'DejaVu Sans',  # 备选字体
                'Arial'  # 最后备选
            ]
            mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

            # 方法2: 清除字体缓存
            self._clear_font_cache()

            # 方法3: 重新构建字体列表
            self._rebuild_font_list()

            # 方法4: 设置所有相关的rc参数
            self._set_all_rc_params()

            # 测试字体设置
            if self._test_chinese_display():
                self.setup_complete = True
                logger.info("✅ 中文字体设置成功")
                return True
            else:
                logger.warning("⚠️ 字体设置可能未完全生效")
                return False

        except Exception as e:
            logger.error(f"❌ 字体设置失败: {e}")
            return False

    def _clear_font_cache(self):
        """清除字体缓存"""
        try:
            # 获取缓存目录
            cache_dir = mpl.get_cachedir()
            import shutil
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                logger.info(f"已清除字体缓存: {cache_dir}")
        except Exception as e:
            logger.warning(f"清除缓存失败: {e}")

    def _rebuild_font_list(self):
        """重新构建字体列表"""
        try:
            # 重新加载字体
            fm._rebuild()
            logger.info("字体列表重建完成")
        except:
            logger.warning("字体列表重建失败，使用现有字体")

    def _set_all_rc_params(self):
        """设置所有相关的rc参数"""
        # 字体相关设置
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 字体大小设置
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16

        logger.info("所有rc参数设置完成")

    def _test_chinese_display(self):
        """测试中文显示"""
        try:
            # 创建测试图表
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([1, 2, 3], [1, 2, 3])
            ax.set_title('中文测试标题 - Chinese Test Title')
            ax.set_xlabel('X轴标签')
            ax.set_ylabel('Y轴标签')

            # 保存测试图片
            test_path = 'chinese_display_test.png'
            plt.savefig(test_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"中文显示测试完成，图片保存为: {test_path}")
            return True

        except Exception as e:
            logger.error(f"中文显示测试失败: {e}")
            return False

    def force_chinese_in_plot(self, fig=None, ax=None):
        """在具体图表中强制使用中文"""
        if fig is None:
            fig = plt.gcf()
        if ax is None:
            ax = plt.gca()

        # 为特定图表设置字体
        for text in fig.texts + ax.texts:
            try:
                text.set_fontfamily('SimHei')
            except:
                pass


# 创建全局字体设置实例
font_setup = FontSetup()


def setup_global_chinese_font():
    """全局中文字体设置函数"""
    return font_setup.setup_chinese_font()