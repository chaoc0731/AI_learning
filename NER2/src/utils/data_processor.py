import torch
from collections import defaultdict
import os


class DataProcessor:
    def __init__(self):
        # 使用普通的字典而不是defaultdict的lambda
        self.word_to_ix = {'[UNK]': 0, '[PAD]': 1}
        self.tag_to_ix = {}
        self.ix_to_tag = {}

        # 初始化基本标签
        base_tags = [
            'O',
            'B-prov', 'I-prov', 'E-prov',
            'B-city', 'I-city', 'E-city',
            'B-district', 'I-district', 'E-district',
            'B-community', 'I-community', 'E-community',
            'B-town', 'I-town', 'E-town',
            'B-poi', 'I-poi', 'E-poi',
            'B-road', 'I-road', 'E-road',
            'B-roadno', 'I-roadno', 'E-roadno',
            'B-subpoi', 'I-subpoi', 'E-subpoi',
            'B-devzone', 'I-devzone', 'E-devzone',
            'B-houseno', 'I-houseno', 'E-houseno',
            'B-intersection', 'I-intersection', 'E-intersection'
        ]
        for tag in base_tags:
            if tag not in self.tag_to_ix:
                self.tag_to_ix[tag] = len(self.tag_to_ix)

    def _get_word_index(self, word):
        """获取单词索引，如果不存在则添加"""
        if word not in self.word_to_ix:
            self.word_to_ix[word] = len(self.word_to_ix)
        return self.word_to_ix[word]

    def _get_tag_index(self, tag):
        """获取标签索引，如果不存在则添加"""
        if tag not in self.tag_to_ix:
            self.tag_to_ix[tag] = len(self.tag_to_ix)
        return self.tag_to_ix[tag]

    def load_data(self, file_path):
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        sentences = []
        labels = []

        with open(file_path, 'r', encoding='utf-8') as f:
            sentence = []
            label = []
            for line in f:
                line = line.strip()
                if not line:
                    if sentence:
                        sentences.append(sentence)
                        labels.append(label)
                        sentence = []
                        label = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        word, tag = parts[0], parts[1]
                        sentence.append(word)
                        label.append(tag)
                        # 构建词典
                        self._get_word_index(word)
                        self._get_tag_index(tag)

            if sentence:
                sentences.append(sentence)
                labels.append(label)

        # 构建索引到标签的映射
        self.ix_to_tag = {ix: tag for tag, ix in self.tag_to_ix.items()}

        print(f"词汇表大小: {len(self.word_to_ix)}")
        print(f"标签数量: {len(self.tag_to_ix)}")
        print(f"样本数量: {len(sentences)}")

        return sentences, labels

    def prepare_sequence(self, words):
        return torch.tensor([self.word_to_ix.get(w, 0) for w in words], dtype=torch.long)

    def prepare_tags(self, tags):
        return torch.tensor([self.tag_to_ix[t] for t in tags], dtype=torch.long)