import torch
import torch.optim as optim
from models.lstm_crf_model import BiLSTM_CRF
from utils.data_processor import DataProcessor
import os


def train_lstm_crf():
    # 数据文件路径
    data_path = "data/train.txt"

    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        possible_paths = [
            "./data/train.txt",
            "../data/train.txt",
            "../../data/train.txt"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        else:
            raise FileNotFoundError(f"找不到数据文件: {data_path}")

    # 数据准备
    processor = DataProcessor()
    train_sentences, train_labels = processor.load_data(data_path)

    print(f"词汇表大小: {len(processor.word_to_ix)}")
    print(f"标签数量: {len(processor.tag_to_ix)}")
    print(f"训练样本数: {len(train_sentences)}")

    # 构建模型
    model = BiLSTM_CRF(
        vocab_size=len(processor.word_to_ix),
        tag_to_ix=processor.tag_to_ix,
        embedding_dim=100,
        hidden_dim=256
    )

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练
    for epoch in range(50):
        total_loss = 0
        for sentence, tags in zip(train_sentences, train_labels):
            model.zero_grad()

            sentence_in = processor.prepare_sequence(sentence).unsqueeze(0)
            targets = processor.prepare_tags(tags).unsqueeze(0)
            mask = torch.ones_like(sentence_in).bool()

            loss = model(sentence_in, targets, mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = total_loss / len(train_sentences)
            print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}')

    # 保存模型 - 使用可序列化的字典
    model_save_path = 'models/lstm_crf_model.pth'
    os.makedirs('models', exist_ok=True)

    save_data = {
        'model_state_dict': model.state_dict(),
        'word_to_ix': dict(processor.word_to_ix),  # 转换为普通字典
        'tag_to_ix': dict(processor.tag_to_ix),  # 转换为普通字典
        'ix_to_tag': dict(processor.ix_to_tag)  # 转换为普通字典
    }

    torch.save(save_data, model_save_path)
    print(f"模型已保存到: {model_save_path}")


if __name__ == '__main__':
    train_lstm_crf()