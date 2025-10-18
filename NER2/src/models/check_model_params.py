import torch
import os


def check_model_parameters():
    """检查模型参数"""
    model_path = "G:\\Pycharm\\Project\\AI_learning\\20231118\\NLPProject05\\NER2\\src\\train\\models\\lstm_crf_model.pth"

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    checkpoint = torch.load(model_path, map_location='cpu')

    print("模型检查点信息:")
    print(f"包含的键: {list(checkpoint.keys())}")

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("\n模型参数详情:")
        for key, value in state_dict.items():
            print(f"  {key}: {value.shape}")

        # 推断模型参数
        if 'word_embeds.weight' in state_dict:
            embedding_dim = state_dict['word_embeds.weight'].shape[1]
            print(f"\n推断的参数:")
            print(f"  - embedding_dim: {embedding_dim}")

        if 'lstm.weight_ih_l0' in state_dict:
            hidden_dim = state_dict['lstm.weight_ih_l0'].shape[0] * 2
            print(f"  - hidden_dim: {hidden_dim}")

    if 'word_to_ix' in checkpoint:
        print(f"\n词汇表大小: {len(checkpoint['word_to_ix'])}")

    if 'tag_to_ix' in checkpoint:
        print(f"标签数量: {len(checkpoint['tag_to_ix'])}")
        print(f"标签: {list(checkpoint['tag_to_ix'].keys())}")


if __name__ == "__main__":
    check_model_parameters()