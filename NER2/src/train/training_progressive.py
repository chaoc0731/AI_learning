import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from models.bert_lstm_crf_model import BertBiLSTM_CRF
from utils.data_processor import DataProcessor
from NER2.src.train.train_bert_lstm_crf_optimized import OptimizedNERDataset
import os
import numpy as np


def train_progressive():
    """渐进式训练：先在小数据上快速验证，再在全数据上训练"""
    local_bert_path = "G:\\Pycharm\\Project\\dataroot\\models\\bert-base-chinese"
    data_path = "data/train.txt"

    # 加载数据
    processor = DataProcessor()
    all_sentences, all_labels = processor.load_data(data_path)

    print(f"总数据量: {len(all_sentences)} 条")

    # 阶段1: 小数据快速验证 (1-2分钟)
    print("\n=== 阶段1: 快速验证 ===")
    quick_sentences = all_sentences[:200]  # 前200条
    quick_labels = all_labels[:200]

    tokenizer = BertTokenizerFast.from_pretrained(local_bert_path)
    model = BertBiLSTM_CRF(local_bert_path, processor.tag_to_ix)

    # 快速训练配置
    quick_dataloader = DataLoader(
        OptimizedNERDataset(quick_sentences, quick_labels, tokenizer),
        batch_size=8,
        shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(2):  # 只训练2个epoch
        total_loss = 0
        for batch_idx, batch in enumerate(quick_dataloader):
            loss = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['labels'],
                batch['mask']
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'快速训练 - Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(quick_dataloader)
        print(f'快速训练 - Epoch {epoch} 完成, 平均Loss: {avg_loss:.4f}')

    # 保存快速验证模型
    quick_model_path = "models/bert_lstm_crf_quick.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'tag_to_ix': dict(processor.tag_to_ix),
        'bert_path': local_bert_path
    }, quick_model_path)
    print(f"快速验证模型保存到: {quick_model_path}")

    # 阶段2: 全数据训练 (如果需要)
    user_input = input("\n是否继续全数据训练? (y/n): ")
    if user_input.lower() == 'y':
        print("\n=== 阶段2: 全数据训练 ===")

        full_dataloader = DataLoader(
            OptimizedNERDataset(all_sentences, all_labels, tokenizer),
            batch_size=16,
            shuffle=True
        )

        # 继续训练
        for epoch in range(3):  # 再训练3个epoch
            total_loss = 0
            for batch_idx, batch in enumerate(full_dataloader):
                loss = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['labels'],
                    batch['mask']
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % 50 == 0:
                    print(f'全数据训练 - Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

            avg_loss = total_loss / len(full_dataloader)
            print(f'全数据训练 - Epoch {epoch} 完成, 平均Loss: {avg_loss:.4f}')

        # 保存最终模型
        final_model_path = "models/bert_lstm_crf_full.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'tag_to_ix': dict(processor.tag_to_ix),
            'bert_path': local_bert_path
        }, final_model_path)
        print(f"全数据模型保存到: {final_model_path}")


if __name__ == '__main__':
    train_progressive()