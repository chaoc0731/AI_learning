import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, get_linear_schedule_with_warmup
from models.bert_lstm_crf_model import BertBiLSTM_CRF
from utils.data_processor import DataProcessor
import os
import math


class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        processor = DataProcessor()
        self.tag_to_ix = processor.tag_to_ix
        self.ix_to_tag = processor.ix_to_tag

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
            return_offsets_mapping=True
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None

        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                aligned_labels.append(0)
            elif word_idx != previous_word_idx:
                aligned_labels.append(self.tag_to_ix.get(labels[word_idx], 0))
            else:
                aligned_labels.append(self.tag_to_ix.get(labels[word_idx], 0))
            previous_word_idx = word_idx

        while len(aligned_labels) < self.max_len:
            aligned_labels.append(0)
        aligned_labels = aligned_labels[:self.max_len]

        mask = []
        for i, word_id in enumerate(word_ids):
            if word_id is not None:
                mask.append(1)
            else:
                if i == 0:
                    mask.append(1)
                else:
                    mask.append(0)

        while len(mask) < self.max_len:
            mask.append(0)
        mask = mask[:self.max_len]

        if len(mask) > 0 and mask[0] == 0:
            mask[0] = 1

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool)
        }


def train_with_gradient_accumulation():
    """使用梯度累积训练"""
    local_bert_path = "G:\\Pycharm\\Project\\dataroot\\models\\bert-base-chinese"
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

    print(f"数据: {len(train_sentences)} 个样本")
    print(f"标签: {len(processor.tag_to_ix)} 个类别")

    # 初始化
    tokenizer = BertTokenizerFast.from_pretrained(local_bert_path)
    model = BertBiLSTM_CRF(local_bert_path, processor.tag_to_ix, dropout_rate=0.3)

    # 数据集 - 使用较大的batch_size
    dataset = NERDataset(train_sentences, train_labels, tokenizer)

    # 梯度累积参数
    effective_batch_size = 16  # 有效batch大小
    batch_size = 4  # 实际batch大小
    accumulation_steps = effective_batch_size // batch_size  # 累积步数

    print(f"梯度累积配置:")
    print(f"  实际batch_size: {batch_size}")
    print(f"  累积步数: {accumulation_steps}")
    print(f"  有效batch_size: {effective_batch_size}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # 学习率调度器
    total_steps = len(dataloader) * 10 // accumulation_steps  # 10个epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # 训练
    model.train()
    global_step = 0

    for epoch in range(10):
        total_loss = 0
        batch_count = 0
        optimizer.zero_grad()  # 在epoch开始时清零梯度

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            mask = batch['mask']

            mask = mask.bool()

            if not mask[:, 0].any():
                continue

            loss = model(input_ids, attention_mask, labels, mask)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            # 梯度累积：除以累积步数
            loss = loss / accumulation_steps
            loss.backward()

            total_loss += loss.item() * accumulation_steps  # 恢复原始loss值用于记录
            batch_count += 1

            # 每accumulation_steps步更新一次参数
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                print(f'Epoch: {epoch}, Step: {global_step}, Loss: {loss.item() * accumulation_steps:.6f}')

        # 处理最后一个不完整的累积批次
        if len(dataloader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f'Epoch {epoch} 完成, 平均Loss: {avg_loss:.6f}')

    # 保存模型
    model_save_path = "models/bert_lstm_crf_accumulated.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'tag_to_ix': dict(processor.tag_to_ix),
        'word_to_ix': dict(processor.word_to_ix),
        'bert_path': local_bert_path
    }, model_save_path)
    print(f"模型已保存到: {model_save_path}")


if __name__ == '__main__':
    train_with_gradient_accumulation()