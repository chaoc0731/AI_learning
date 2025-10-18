import torch
import os


def check_lstm_model():
    """检查LSTM模型状态"""
    model_path = "G:\\Pycharm\\Project\\AI_learning\\20231118\\NLPProject05\\NER2\\src\\train\\models\\lstm_crf_model.pth"

    if not os.path.exists(model_path):
        print(f"✗ LSTM模型文件不存在: {model_path}")
        return

    print("检查LSTM模型...")
    checkpoint = torch.load(model_path, map_location='cpu')

    print("模型检查点信息:")
    print(f"包含的键: {list(checkpoint.keys())}")

    # 检查词汇表
    if 'word_to_ix' in checkpoint:
        word_to_ix = checkpoint['word_to_ix']
        print(f"\n词汇表大小: {len(word_to_ix)}")
        print("前10个词汇:", list(word_to_ix.keys())[:10])

    # 检查标签
    if 'tag_to_ix' in checkpoint:
        tag_to_ix = checkpoint['tag_to_ix']
        print(f"\n标签数量: {len(tag_to_ix)}")
        print("所有标签:", tag_to_ix)

    # 检查模型参数
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\n模型参数:")
        for key, value in state_dict.items():
            print(f"  {key}: {value.shape}")

        # 检查参数是否正常（没有NaN或无穷大）
        has_nan = False
        has_inf = False
        for key, value in state_dict.items():
            if torch.isnan(value).any():
                print(f"  ✗ {key} 包含NaN值")
                has_nan = True
            if torch.isinf(value).any():
                print(f"  ✗ {key} 包含无穷大值")
                has_inf = True

        if not has_nan and not has_inf:
            print("  ✓ 所有参数正常")

    # 测试模型预测
    test_model_prediction(checkpoint)


def test_model_prediction(checkpoint):
    """测试模型预测能力"""
    print("\n测试模型预测...")

    from models.lstm_crf_model import load_lstm_model_simple

    try:
        model, _ = load_lstm_model_simple(
            "G:\\Pycharm\\Project\\AI_learning\\20231118\\NLPProject05\\NER2\\src\\train\\models\\lstm_crf_model.pth"
        )

        # 测试简单文本
        test_texts = [
            "李明在北京",
            "清华大学",
            "阿里巴巴公司",
            "上海市政府"
        ]

        word_to_ix = checkpoint['word_to_ix']
        ix_to_tag = checkpoint['ix_to_tag']

        for text in test_texts:
            print(f"\n测试文本: '{text}'")
            words = list(text)

            # 准备输入
            sentence_in = torch.tensor([word_to_ix.get(w, 0) for w in words]).unsqueeze(0)
            mask = torch.ones_like(sentence_in).bool()

            with torch.no_grad():
                tags = model(sentence_in, mask=mask)

            # 转换标签
            predicted_tags = [ix_to_tag.get(tag_idx, 'O') for tag_idx in tags[0][:len(words)]]
            print(f"预测标签: {predicted_tags}")

            # 提取实体
            entities = []
            current_entity = None

            for i, (word, tag) in enumerate(zip(words, predicted_tags)):
                if tag.startswith('B-'):
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {'text': word, 'type': tag[2:], 'start': i}
                elif tag.startswith('I-') and current_entity and current_entity['type'] == tag[2:]:
                    current_entity['text'] += word
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            if current_entity:
                entities.append(current_entity)

            print(f"识别实体: {entities}")

    except Exception as e:
        print(f"模型测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_lstm_model()