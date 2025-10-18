import os


def create_address_training_data():
    """创建地址实体训练数据"""
    training_data = []

    # 地址样本数据
    address_samples = [
        # 完整地址
        ("北京市海淀区中关村大街27号",
         ["B-prov", "E-prov", "B-city", "E-city", "B-district", "E-district", "B-road", "I-road", "I-road", "E-road",
          "B-roadno", "E-roadno"]),

        ("上海市浦东新区陆家嘴金融中心",
         ["B-prov", "E-prov", "B-city", "E-city", "B-district", "I-district", "E-district", "B-poi", "I-poi", "I-poi",
          "E-poi"]),

        ("广州市天河区天河路228号正佳广场",
         ["B-prov", "E-prov", "B-city", "E-city", "B-district", "E-district", "B-road", "E-road", "B-roadno",
          "I-roadno", "E-roadno", "B-poi", "I-poi", "E-poi"]),

        # 简单地址
        ("北京大学", ["B-poi", "I-poi", "E-poi"]),
        ("清华大学", ["B-poi", "I-poi", "E-poi"]),
        ("阿里巴巴", ["B-poi", "I-poi", "I-poi", "E-poi"]),

        # 道路地址
        ("长安街", ["B-road", "I-road", "E-road"]),
        ("南京东路", ["B-road", "I-road", "E-road"]),
        ("解放大道123号", ["B-road", "I-road", "E-road", "B-roadno", "I-roadno", "E-roadno"]),

        # 行政区划
        ("江苏省南京市", ["B-prov", "E-prov", "B-city", "E-city"]),
        ("浙江省杭州市西湖区", ["B-prov", "E-prov", "B-city", "E-city", "B-district", "E-district"]),
        ("四川省成都市武侯区", ["B-prov", "E-prov", "B-city", "E-city", "B-district", "E-district"]),
    ]

    # 转换为训练格式
    for text, labels in address_samples:
        words = list(text)
        if len(words) == len(labels):
            for word, label in zip(words, labels):
                training_data.append(f"{word} {label}")
            training_data.append("")  # 空行分隔

    # 写入文件
    data_path = "data/train.txt"
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    with open(data_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(training_data))

    print(f"创建了 {len(address_samples)} 个地址训练样本")
    print(f"数据已保存到: {data_path}")


if __name__ == "__main__":
    create_address_training_data()