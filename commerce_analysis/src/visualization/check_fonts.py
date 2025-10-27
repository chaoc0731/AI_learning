
"""
字体检查脚本
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def check_chinese_fonts():
    """检查系统中可用的中文字体"""
    print("=" * 50)
    print("系统中可用的中文字体")
    print("=" * 50)

    # 获取所有字体
    fonts = [f.name for f in fm.fontManager.ttflist]

    # 常见的中文字体
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'STSong', 'KaiTi', 'FangSong',
        'SimSun', 'NSimSun', 'SimHei', 'YouYuan', 'STKaiti'
    ]

    available_chinese_fonts = []
    for font in chinese_fonts:
        if any(font in f for f in fonts):
            available_chinese_fonts.append(font)
            print(f"✅ {font}")
        else:
            print(f"❌ {font}")

    print(f"\n找到 {len(available_chinese_fonts)} 种中文字体")

    # 测试显示中文
    print("\n测试中文显示...")
    plt.rcParams['font.sans-serif'] = available_chinese_fonts + ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 4))
    plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
    plt.title('中文标题测试 - Chinese Title Test')
    plt.xlabel('X轴标签')
    plt.ylabel('Y轴标签')
    plt.tight_layout()
    plt.savefig('chinese_test.png', dpi=100, bbox_inches='tight')
    plt.show()

    print("测试图表已保存为 'chinese_test.png'")


if __name__ == "__main__":
    check_chinese_fonts()