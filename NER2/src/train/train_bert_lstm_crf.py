import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig
from models.bert_lstm_crf_model import BertBiLSTM_CRF
from utils.data_processor import DataProcessor
import os
import copy


class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        # self.tag_to_ix = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6}
        # self.tag_to_ix = {'O': 0,
        #                   'B-prov': 1, 'I-prov': 2, 'E-prov': 3,
        #                   'B-city': 4, 'I-city': 5, 'E-city': 6,
        #                   'B-district': 7, 'I-district': 8, 'E-district': 9,
        #                   'B-community': 10, 'I-community': 11, 'E-community': 12,
        #                   'B-town': 13, 'I-town': 14, 'E-town': 15,
        #                   'B-poi': 16, 'I-poi': 17, 'E-poi': 18,
        #                   'B-road': 19, 'I-road': 20, 'E-road': 21,
        #                   'B-roadno': 22, 'I-roadno': 23, 'E-roadno': 24,
        #                   'B-subpoi': 25, 'I-subpoi': 26, 'E-subpoi': 27,
        #                   'B-devzone': 28, 'I-devzone': 29, 'E-devzone': 30,
        #                   'B-houseno': 31, 'I-houseno': 32, 'E-houseno': 33,
        #                   'B-intersection': 34, 'I-intersection': 35, 'E-intersection': 36,
        #                   }
        # self.ix_to_tag = {v: k for k, v in self.tag_to_ix.items()}
        # 使用数据处理器中的标签映射
        processor = DataProcessor()
        self.tag_to_ix = processor.tag_to_ix
        self.ix_to_tag = processor.ix_to_tag

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

        for word_idx in word_ids:
            if word_idx is None:
                # 特殊token ([CLS], [SEP], [PAD]) 设置为0（O标签）
                aligned_labels.append(0)  # 使用O标签而不是-100
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

        # 创建mask，特殊token位置为False
        mask = [1 if word_id is not None else 0 for word_id in word_ids]
        while len(mask) < self.max_len:
            mask.append(0)
        mask = mask[:self.max_len]

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

    # 检查必要的文件是否存在
    required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt']
    for file in required_files:
        if not os.path.exists(os.path.join(bert_path, file)):
            raise FileNotFoundError(f"在 {bert_path} 中找不到 {file}")

    # 加载快速分词器和config
    tokenizer = BertTokenizerFast.from_pretrained(bert_path)
    config = BertConfig.from_pretrained(bert_path)

    return tokenizer, config


def train_bert_lstm_crf():
    # 本地BERT模型路径 - 修改为你的实际路径
    local_bert_path = "G:\\Pycharm\\Project\\dataroot\\models\\bert-base-chinese"

    # 数据文件路径 - 使用相对路径
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

    # 加载数据
    processor = DataProcessor()
    train_sentences, train_labels = processor.load_data(data_path)

    print(f"加载了 {len(train_sentences)} 条训练数据")
    print(f"标签类型: {processor.tag_to_ix}")

    try:
        # 加载本地BERT模型
        print(f"正在从本地加载BERT模型: {local_bert_path}")
        tokenizer, bert_config = load_local_bert_model(local_bert_path)

        # 初始化模型
        model = BertBiLSTM_CRF(local_bert_path, processor.tag_to_ix)
        print("模型初始化完成")

    except Exception as e:
        print(f"加载本地模型失败: {e}")
        print("尝试从网络下载模型...")
        # 备用方案：从网络下载快速分词器
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        model = BertBiLSTM_CRF('bert-base-chinese', processor.tag_to_ix)

    # 准备数据集
    dataset = NERDataset(train_sentences, train_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # 训练
    model.train()
    for epoch in range(3):  # 减少epoch数量用于测试
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

            loss = model(input_ids, attention_mask, labels, mask)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"发现无效的loss值: {loss}, 跳过该batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_idx % 2 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f'Epoch {epoch} 完成, 平均Loss: {avg_loss:.4f}')

    # 保存模型 - 使用可序列化的字典
    model_save_path = "models/bert_lstm_crf_model.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 创建可序列化的字典副本
    save_data = {
        'model_state_dict': model.state_dict(),
        'tag_to_ix': dict(processor.tag_to_ix),  # 转换为普通字典
        'word_to_ix': dict(processor.word_to_ix),  # 转换为普通字典
        'bert_path': local_bert_path
    }

    torch.save(save_data, model_save_path)
    print(f"模型已保存到: {model_save_path}")

    # 同时保存处理器的状态
    processor_save_path = "models/processor_state.pth"
    torch.save({
        'word_to_ix': dict(processor.word_to_ix),
        'tag_to_ix': dict(processor.tag_to_ix),
        'ix_to_tag': dict(processor.ix_to_tag)
    }, processor_save_path)
    print(f"处理器状态已保存到: {processor_save_path}")


if __name__ == '__main__':
    train_bert_lstm_crf()