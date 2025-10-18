import os
import torch


def check_model_files():
    print("检查模型文件...")

    # 检查BERT模型
    bert_model_path = "models/bert_lstm_crf_model.pth"
    if os.path.exists(bert_model_path):
        print(f"✓ BERT模型文件存在: {bert_model_path}")
        try:
            checkpoint = torch.load(bert_model_path, map_location='cpu')
            print(f"  - 包含的键: {list(checkpoint.keys())}")
            print(f"  - 标签数量: {len(checkpoint.get('tag_to_ix', {}))}")
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
    else:
        print(f"✗ BERT模型文件不存在: {bert_model_path}")

    # 检查LSTM模型
    lstm_model_path = "models/lstm_crf_model.pth"
    if os.path.exists(lstm_model_path):
        print(f"✓ LSTM模型文件存在: {lstm_model_path}")
        try:
            checkpoint = torch.load(lstm_model_path, map_location='cpu')
            print(f"  - 包含的键: {list(checkpoint.keys())}")
            print(f"  - 词汇表大小: {len(checkpoint.get('word_to_ix', {}))}")
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
    else:
        print(f"✗ LSTM模型文件不存在: {lstm_model_path}")

    # 检查目录结构
    print("\n检查目录结构:")
    for root, dirs, files in os.walk("."):
        if "model" in root.lower() or ".pth" in str(files):
            rel_path = os.path.relpath(root)
            print(f"目录: {rel_path}")
            for file in files:
                if file.endswith('.pth'):
                    print(f"  - {file}")


if __name__ == "__main__":
    check_model_files()