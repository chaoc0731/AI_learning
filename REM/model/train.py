# model/train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup, BertTokenizer
import json
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from model import RelationExtractionModel
from REM.config import Config


class RelationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, label_map):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = self.label_map[item["relation"]]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_model():
    config = Config()

    # 加载数据
    with open(f"{config.DATA_DIR}/processed_data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建标签映射
    label_map = {label: idx for idx, label in enumerate(config.RELATION_TYPES)}

    # 保存标签映射
    with open(f"{config.DATA_DIR}/label_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # 分割数据集
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # 初始化tokenizer和模型（使用本地模型）
    tokenizer = BertTokenizer.from_pretrained(config.get_model_path())
    model = RelationExtractionModel(config)

    # 创建数据加载器
    train_dataset = RelationDataset(train_data, tokenizer, config.MAX_LENGTH, label_map)
    val_dataset = RelationDataset(val_data, tokenizer, config.MAX_LENGTH, label_map)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.to(device)

    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        print(f'Train Loss: {total_loss / len(train_loader):.4f}')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}')
        print(f'Val Accuracy: {100 * correct / total:.2f}%')
        print('-' * 50)

    # 保存模型
    model_save_path = f"{config.MODEL_DIR}/checkpoint/model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"模型训练完成并保存到: {model_save_path}")


if __name__ == "__main__":
    train_model()