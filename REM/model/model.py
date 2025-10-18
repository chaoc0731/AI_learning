# model/model.py
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import os
from REM.config import Config


class RelationExtractionModel(nn.Module):
    def __init__(self, config: Config):
        super(RelationExtractionModel, self).__init__()

        # 使用本地模型路径
        model_path = config.get_model_path()
        print(f"加载模型从: {model_path}")

        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(config.RELATION_TYPES))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class RelationPredictor:
    def __init__(self, model_path: str, config: Config):
        self.config = config

        # 使用本地BERT模型
        bert_path = config.get_model_path()
        print(f"加载tokenizer从: {bert_path}")
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        # 加载关系抽取模型
        self.model = RelationExtractionModel(config)

        # 检查模型文件是否存在
        if os.path.exists(model_path):
            print(f"加载关系抽取模型从: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            print(f"警告: 关系抽取模型文件不存在 {model_path}，使用未训练模型")

        self.model.eval()

    def predict(self, text: str, entity1: str, entity2: str) -> dict:
        """预测实体关系"""
        # 准备输入
        processed_text = f"{text}[SEP]{entity1}[SEP]{entity2}"

        inputs = self.tokenizer(
            processed_text,
            max_length=self.config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()

        return {
            "relation": self.config.RELATION_TYPES[predicted_class],
            "confidence": confidence,
            "text": text,
            "entity1": entity1,
            "entity2": entity2
        }