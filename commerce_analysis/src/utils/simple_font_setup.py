"""
简单字体设置 - 避免复杂的字体重建
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import logging

logger = logging.getLogger(__name__)


def setup_chinese_font_simple():
    """简单的中文字体设置"""
    try:
        # 方法1: 直接设置rcParams
        plt.rcParams['font.sans-serif'] = [
            'SimHei',  # 黑体 (Windows)
            'Microsoft YaHei',  # 微软雅黑 (Windows)
            'DejaVu Sans',  # 备选字体
            'Arial Unicode MS',  # macOS/Linux
            'Arial'  # 最后备选
        ]
        plt.rcParams['axes.unicode_minus'] = False

        # 方法2: 设置字体大小
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

        logger.info("✅ 简单字体设置完成")
        return True

    except Exception as e:
        logger.warning(f"简单字体设置警告: {e}")
        return False


def test_chinese_display():
    """测试中文显示"""
    try:
        # 创建测试图表
        fig, ax = plt.subplots(figsize=(8, 6))

        # 测试数据
        categories = ['产品A', '产品B', '产品C', '产品D']
        values = [23, 45, 56, 78]

        # 创建柱状图
        bars = ax.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])

        # 设置中文标题和标签
        ax.set_title('中文显示测试 - 电商数据分析', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('产品类别', fontsize=12)
        ax.set_ylabel('销售数量', fontsize=12)

        # 在柱子上显示数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{value}', ha='center', va='bottom', fontsize=10)

        # 添加网格
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # 保存测试图片
        plt.savefig('chinese_display_test.png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.show()

        print("✅ 中文显示测试完成！图片已保存为 'chinese_display_test.png'")
        print("请检查图片中的中文是否正常显示")
        return True

    except Exception as e:
        print(f"❌ 中文显示测试失败: {e}")
        return False


if __name__ == "__main__":
    setup_chinese_font_simple()
    test_chinese_display()