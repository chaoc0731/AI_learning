import torch
from transformers import BertTokenizerFast
from models.bert_lstm_crf_model import BertBiLSTM_CRF
from utils.data_processor import DataProcessor
import os


def debug_training_issue():
    """诊断训练问题"""
    print("诊断BERT训练问题...")

    # 加载数据
    processor = DataProcessor()
    data_path = "data/train.txt"

    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        # 尝试其他可能的路径
        possible_paths = [
            "./data/train.txt",
            "../data/train.txt",
            "../../data/train.txt",
            "NER2/data/train.txt"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        else:
            raise FileNotFoundError(f"找不到数据文件，请检查路径。尝试的路径: {possible_paths}")

    print(f"使用数据文件: {data_path}")

    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return

    train_sentences, train_labels = processor.load_data(data_path)
    print(f"数据统计: {len(train_sentences)} 条样本")
    print(f"标签数量: {len(processor.tag_to_ix)}")
    print(f"标签: {list(processor.tag_to_ix.keys())}")

    # 检查数据样本
    print("\n检查前3个样本:")
    for i in range(min(3, len(train_sentences))):
        print(f"样本 {i}:")
        print(f"  句子: {''.join(train_sentences[i])}")
        print(f"  标签: {train_labels[i]}")
        print(f"  长度: {len(train_sentences[i])}")

    # 初始化模型
    local_bert_path = "G:\\Pycharm\\Project\\dataroot\\models\\bert-base-chinese"
    tokenizer = BertTokenizerFast.from_pretrained(local_bert_path)
    model = BertBiLSTM_CRF(local_bert_path, processor.tag_to_ix)

    # 测试单个样本的前向传播
    print("\n测试单个样本前向传播:")
    test_sentence = train_sentences[0]
    test_labels = train_labels[0]

    # 编码输入
    encoding = tokenizer(
        test_sentence,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt',
        return_offsets_mapping=True
    )

    # 准备标签
    word_ids = encoding.word_ids(batch_index=0)
    aligned_labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(0)  # O标签
        elif word_idx != previous_word_idx:
            aligned_labels.append(processor.tag_to_ix.get(test_labels[word_idx], 0))
        else:
            aligned_labels.append(processor.tag_to_ix.get(test_labels[word_idx], 0))
        previous_word_idx = word_idx

    # 确保标签长度
    while len(aligned_labels) < 128:
        aligned_labels.append(0)
    aligned_labels = aligned_labels[:128]

    labels_tensor = torch.tensor(aligned_labels, dtype=torch.long).unsqueeze(0)

    print(f"输入形状: {encoding['input_ids'].shape}")
    print(f"标签形状: {labels_tensor.shape}")

    # 前向传播
    model.eval()
    with torch.no_grad():
        loss = model(
            encoding['input_ids'],
            encoding['attention_mask'],
            labels_tensor
        )

    print(f"测试样本loss: {loss.item()}")

    # 检查模型参数
    print("\n检查模型参数:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape} - 可训练")
            total_params += param.numel()
        else:
            print(f"  {name}: {param.shape} - 冻结")

    print(f"总可训练参数: {total_params:,}")

    # 检查梯度
    print("\n检查梯度:")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    optimizer.zero_grad()

    loss = model(
        encoding['input_ids'],
        encoding['attention_mask'],
        labels_tensor
    )

    loss.backward()

    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                has_grad = True
                print(f"  {name}: 梯度范数 = {grad_norm:.6f}")

    if not has_grad:
        print("  警告: 所有参数梯度都为0!")


if __name__ == "__main__":
    debug_training_issue()