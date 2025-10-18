import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig
from models.bert_lstm_crf_model import BertBiLSTM_CRF
from utils.data_processor import DataProcessor
import os
import numpy as np


class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 使用数据处理器中的标签映射
        processor = DataProcessor()
        self.tag_to_ix = processor.tag_to_ix
        self.ix_to_tag = processor.ix_to_tag

        print(f"数据集: {len(sentences)} 个样本")
        print(f"标签数量: {len(self.tag_to_ix)}")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]

        # 使用快速分词器进行编码
        encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
            return_offsets_mapping=True
        )

        # 获取word_ids（只有快速分词器支持）
        word_ids = encoding.word_ids(batch_index=0)

        # 对齐标签
        aligned_labels = []
        previous_word_idx = None

        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                # 特殊token ([CLS], [SEP], [PAD]) 设置为0（O标签）
                aligned_labels.append(0)  # 使用O标签
            elif word_idx != previous_word_idx:
                # 当前词的第一个子词
                aligned_labels.append(self.tag_to_ix.get(labels[word_idx], 0))
            else:
                # 当前词的其他子词，使用相同的标签
                aligned_labels.append(self.tag_to_ix.get(labels[word_idx], 0))
            previous_word_idx = word_idx

        # 确保标签长度与输入一致
        while len(aligned_labels) < self.max_len:
            aligned_labels.append(0)  # 使用O标签填充
        aligned_labels = aligned_labels[:self.max_len]

        # 创建mask：实际token为True，特殊token为False
        # 但需要确保第一个时间步（[CLS]）为True
        mask = []
        for i, word_id in enumerate(word_ids):
            if word_id is not None:
                mask.append(1)  # 实际token
            else:
                # 特殊token，但第一个token（[CLS]）需要为True
                if i == 0:  # 第一个token是[CLS]
                    mask.append(1)
                else:  # 其他特殊token（[SEP], [PAD]）
                    mask.append(0)

        # 填充mask
        while len(mask) < self.max_len:
            mask.append(0)
        mask = mask[:self.max_len]

        # 双重检查：确保第一个时间步为True
        if len(mask) > 0 and mask[0] == 0:
            print(f"警告: 第一个时间步mask为0，强制设置为1")
            mask[0] = 1

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool)
        }


def load_local_bert_model(bert_path):
    """加载本地BERT模型和快速分词器"""
    if not os.path.exists(bert_path):
        raise FileNotFoundError(f"BERT模型路径不存在: {bert_path}")

    required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt']
    for file in required_files:
        if not os.path.exists(os.path.join(bert_path, file)):
            raise FileNotFoundError(f"在 {bert_path} 中找不到 {file}")

    tokenizer = BertTokenizerFast.from_pretrained(bert_path)
    config = BertConfig.from_pretrained(bert_path)

    return tokenizer, config


def validate_dataset(dataloader):
    """验证数据集mask和标签"""
    print("验证数据集mask和标签...")
    for i, batch in enumerate(dataloader):
        if i >= 2:  # 只检查前2个batch
            break

        mask = batch['mask']
        labels = batch['labels']

        print(f"Batch {i}:")
        print(f"  mask形状: {mask.shape}")
        print(f"  第一个时间步mask: {mask[:, 0]}")
        print(f"  第一个时间步是否有True: {mask[:, 0].any().item()}")

        # 检查mask中True的数量
        true_count = mask.sum(dim=1)
        print(f"  每个样本的有效token数: {true_count.tolist()}")

        # 检查标签分布
        for j in range(min(2, mask.shape[0])):  # 检查前2个样本
            sample_mask = mask[j]
            sample_labels = labels[j]

            # 只显示有效token的标签
            valid_indices = sample_mask.nonzero(as_tuple=True)[0]
            valid_labels = sample_labels[valid_indices]

            print(f"  样本{j}的有效标签: {valid_labels.tolist()[:10]}...")  # 只显示前10个


def train_bert_lstm_crf():
    # 本地BERT模型路径
    local_bert_path = "G:\\Pycharm\\Project\\dataroot\\models\\bert-base-chinese"

    # 数据文件路径
    data_path = "data/train.txt"

    # 检查数据文件是否存在
    if not os.path.exists(data_path):
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
            raise FileNotFoundError(f"找不到数据文件: {data_path}")

    print(f"使用数据文件: {data_path}")

    # 加载数据
    processor = DataProcessor()
    train_sentences, train_labels = processor.load_data(data_path)

    print(f"加载了 {len(train_sentences)} 条训练数据")
    print(f"标签类型: {dict(processor.tag_to_ix)}")

    # 检查数据质量
    entity_counts = {}
    for labels in train_labels:
        for label in labels:
            if label != 'O':
                entity_counts[label] = entity_counts.get(label, 0) + 1

    print(f"实体统计: {entity_counts}")

    try:
        # 加载本地BERT模型
        print(f"正在从本地加载BERT模型: {local_bert_path}")
        tokenizer, bert_config = load_local_bert_model(local_bert_path)

        # 初始化模型
        model = BertBiLSTM_CRF(local_bert_path, processor.tag_to_ix, dropout_rate=0.3)
        print("模型初始化完成")

    except Exception as e:
        print(f"加载本地模型失败: {e}")
        print("尝试从网络下载模型...")
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        model = BertBiLSTM_CRF('bert-base-chinese', processor.tag_to_ix, dropout_rate=0.3)

    # 准备数据集
    dataset = NERDataset(train_sentences, train_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 验证数据集
    validate_dataset(dataloader)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # 训练
    model.train()
    for epoch in range(5):
        total_loss = 0
        batch_count = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            mask = batch['mask']

            # 确保mask是正确的布尔类型
            mask = mask.bool()

            # 检查mask有效性
            if not mask[:, 0].any():
                print(f"错误: batch {batch_idx} 第一个时间步mask全为False")
                print(f"  mask: {mask[:, 0]}")
                continue

            loss = model(input_ids, attention_mask, labels, mask)

            # 检查loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"发现无效的loss值: {loss}, 跳过该batch")
                continue

            if loss.item() == 0:
                print(f"警告: loss为0, batch {batch_idx}")

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_idx % 2 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f'Epoch {epoch} 完成, 平均Loss: {avg_loss:.6f}')

    # 保存模型
    model_save_path = "models/bert_lstm_crf_model.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    save_data = {
        'model_state_dict': model.state_dict(),
        'tag_to_ix': dict(processor.tag_to_ix),
        'word_to_ix': dict(processor.word_to_ix),
        'bert_path': local_bert_path
    }

    torch.save(save_data, model_save_path)
    print(f"模型已保存到: {model_save_path}")


if __name__ == '__main__':
    train_bert_lstm_crf()