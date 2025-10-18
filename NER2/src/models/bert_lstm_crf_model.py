import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from TorchCRF import CRF
import os


class BertBiLSTM_CRF(nn.Module):
    def __init__(self, bert_path, tag_to_ix, hidden_dim=256, lstm_layers=2, dropout_rate=0.3):
        super(BertBiLSTM_CRF, self).__init__()

        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        print(f"初始化BERT+BiLSTM+CRF模型:")
        print(f"  标签数量: {self.tagset_size}")
        print(f"  隐藏层维度: {hidden_dim}")
        print(f"  LSTM层数: {lstm_layers}")

        # 加载本地BERT模型
        if os.path.exists(bert_path):
            print(f"从本地加载BERT模型: {bert_path}")
            self.bert = BertModel.from_pretrained(bert_path)
        else:
            print(f"本地模型不存在，从网络下载: {bert_path}")
            self.bert = BertModel.from_pretrained(bert_path)

        # 确保BERT参数可训练
        for param in self.bert.parameters():
            param.requires_grad = True

        # LSTM层
        self.lstm = nn.LSTM(
            self.bert.config.hidden_size,
            hidden_dim // 2,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 输出层
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 初始化输出层权重
        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.constant_(self.hidden2tag.bias, 0.0)

        # CRF层
        self.crf = CRF(self.tagset_size, batch_first=True)

        print("模型初始化完成")

    def forward(self, input_ids, attention_mask, tags=None, mask=None):
        # 获取BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # 应用dropout
        sequence_output = self.dropout(sequence_output)

        # LSTM处理
        lstm_out, _ = self.lstm(sequence_output)
        lstm_out = self.dropout(lstm_out)

        # 线性变换到标签空间
        emissions = self.hidden2tag(lstm_out)

        # 使用attention_mask创建mask
        if mask is None:
            mask = attention_mask.bool()
        else:
            mask = mask.bool()

        # 确保mask与emissions的维度匹配
        if mask.size(1) != emissions.size(1):
            if mask.size(1) < emissions.size(1):
                pad_size = emissions.size(1) - mask.size(1)
                mask = torch.nn.functional.pad(mask, (0, pad_size), value=False)
            else:
                mask = mask[:, :emissions.size(1)]

        # 检查mask是否有效（第一个时间步不能全为0）
        if mask.size(0) > 0 and not mask[:, 0].any():
            print("警告: 第一个时间步的mask全为0，进行调整...")
            # 将第一个时间步设置为True
            mask[:, 0] = True

        if tags is not None:
            # 确保tags在有效范围内
            tags = torch.clamp(tags, 0, self.tagset_size - 1)

            # 计算CRF损失
            try:
                # 检查是否有有效的标签（不是全部都是0）
                valid_labels = (tags != 0).any()
                if not valid_labels:
                    print("警告: 所有标签都是O标签!")

                # 确保mask的第一个时间步不是全False
                if not mask[:, 0].any():
                    print("错误: mask的第一个时间步全为False!")
                    # 创建一个有效的mask
                    batch_size = mask.size(0)
                    seq_len = mask.size(1)
                    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
                    # 只保留实际文本部分
                    for i in range(batch_size):
                        actual_len = attention_mask[i].sum().item()
                        mask[i, actual_len:] = False

                loss = -self.crf(emissions, tags, mask=mask, reduction='mean')

                # 检查loss是否为NaN或0
                if torch.isnan(loss):
                    print("警告: Loss为NaN!")
                    return torch.tensor(1.0, requires_grad=True)
                elif loss.item() == 0:
                    print("警告: Loss为0!")

                return loss
            except Exception as e:
                print(f"CRF损失计算错误: {e}")
                print(f"tags范围: {tags.min()} ~ {tags.max()}, tagset_size: {self.tagset_size}")
                print(f"emissions形状: {emissions.shape}")
                print(f"mask形状: {mask.shape}")
                print(f"mask第一个时间步: {mask[:, 0]}")

                # 返回一个默认损失
                return torch.tensor(1.0, requires_grad=True)
        else:
            # 解码
            try:
                # 检查mask有效性
                if not mask[:, 0].any():
                    print("解码时mask第一个时间步全为False，进行调整")
                    batch_size = mask.size(0)
                    seq_len = mask.size(1)
                    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
                    for i in range(batch_size):
                        actual_len = attention_mask[i].sum().item()
                        mask[i, actual_len:] = False

                return self.crf.decode(emissions, mask=mask)
            except Exception as e:
                print(f"CRF解码错误: {e}")
                # 返回一个默认的预测
                batch_size = emissions.size(0)
                seq_length = emissions.size(1)
                return [torch.zeros(seq_length, dtype=torch.long) for _ in range(batch_size)]