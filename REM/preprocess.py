# preprocess.py
import json
import os
from typing import List, Dict
from config import Config


class DataProcessor:
    def __init__(self, config: Config):
        self.config = config

    def load_data(self, file_path: str) -> List[Dict]:
        """加载原始数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 确保返回的是数据列表而不是整个字典
            if isinstance(data, dict) and 'data' in data:
                return data['data']
            return data

    def convert_to_relation_format(self, data: List[Dict]) -> List[Dict]:
        """转换为关系分类格式"""
        processed_data = []

        for item in data:
            # 确保item是字典类型
            if not isinstance(item, dict):
                print(f"跳过非字典项: {item}")
                continue

            text = item.get("text", "")
            entities = item.get("entities", [])

            if not text or not entities:
                continue

            # 为每对实体创建样本
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    ent1 = entities[i]
                    ent2 = entities[j]

                    # 确保实体是字典类型
                    if not isinstance(ent1, dict) or not isinstance(ent2, dict):
                        continue

                    # 创建模型输入
                    processed_text = f"{text}[SEP]{ent1.get('text', '')}[SEP]{ent2.get('text', '')}"

                    # 确定关系
                    relation = self._determine_relation(item, ent1, ent2)

                    processed_data.append({
                        "text": processed_text,
                        "entity1": ent1.get('text', ''),
                        "entity2": ent2.get('text', ''),
                        "relation": relation
                    })

        return processed_data

    def _determine_relation(self, item: Dict, ent1: Dict, ent2: Dict) -> str:
        """确定实体间关系"""
        # 如果原始数据中已标注关系，则使用标注的关系
        # 否则使用"无关系"
        return item.get("relation", "无关系")

    def save_processed_data(self, data: List[Dict], output_path: str):
        """保存处理后的数据"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    config = Config()
    processor = DataProcessor(config)

    try:
        # 加载和处理数据
        raw_data = processor.load_data(f"{config.DATA_DIR}/raw_data.json")
        print(f"成功加载 {len(raw_data)} 条原始数据")

        processed_data = processor.convert_to_relation_format(raw_data)

        # 保存处理后的数据
        processor.save_processed_data(
            processed_data,
            f"{config.DATA_DIR}/processed_data.json"
        )

        print(f"处理完成，共生成 {len(processed_data)} 个训练样本")

        # 统计关系分布
        relation_count = {}
        for item in processed_data:
            relation = item["relation"]
            relation_count[relation] = relation_count.get(relation, 0) + 1

        print("关系分布统计:")
        for rel, count in relation_count.items():
            print(f"  {rel}: {count} 条")

    except Exception as e:
        print(f"处理数据时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()