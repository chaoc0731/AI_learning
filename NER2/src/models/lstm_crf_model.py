import torch
import torch.nn as nn
from TorchCRF import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim=100, hidden_dim=256, lstm_layers=2):
        """
        Args:
            vocab_size: 词汇表大小
            tag_to_ix: 标签到索引的映射
            embedding_dim: 词向量维度
            hidden_dim: LSTM隐藏层维度（双向，所以每边是hidden_dim//2）
            lstm_layers: LSTM层数
        """
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # 词嵌入层
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        # LSTM层 - 双向，所以输出维度是 hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=lstm_layers, bidirectional=True, batch_first=True)

        # 输出层
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # CRF层
        self.crf = CRF(self.tagset_size, batch_first=True)

    def forward(self, sentence, tags=None, mask=None):
        # 词嵌入
        embeds = self.word_embeds(sentence)

        # LSTM处理
        lstm_out, _ = self.lstm(embeds)

        # 线性变换到标签空间
        emissions = self.hidden2tag(lstm_out)

        if tags is not None:
            # 计算CRF损失
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            # 解码
            return self.crf.decode(emissions, mask=mask)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cpu'):
        """从检查点加载模型，手动设置正确参数"""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 从检查点推断参数
        word_embeds_weight = checkpoint['model_state_dict']['word_embeds.weight']
        embedding_dim = word_embeds_weight.size(1)
        vocab_size = word_embeds_weight.size(0)

        # 根据权重形状推断LSTM参数
        lstm_weight_ih_l0 = checkpoint['model_state_dict']['lstm.weight_ih_l0']

        # LSTM的输入门、遗忘门、细胞门、输出门各占1/4
        # 对于双向LSTM，每边的隐藏维度是总权重数的1/8
        hidden_dim_per_direction = lstm_weight_ih_l0.size(0) // 4
        hidden_dim = hidden_dim_per_direction * 2  # 双向

        # 推断LSTM层数
        lstm_layers = 0
        for key in checkpoint['model_state_dict'].keys():
            if key.startswith('lstm.weight_ih_l') and not key.endswith('_reverse'):
                lstm_layers = max(lstm_layers, int(key.split('_l')[-1]) + 1)

        tag_to_ix = checkpoint['tag_to_ix']

        print(f"从检查点推断参数:")
        print(f"  - vocab_size: {vocab_size}")
        print(f"  - embedding_dim: {embedding_dim}")
        print(f"  - hidden_dim: {hidden_dim}")
        print(f"  - lstm_layers: {lstm_layers}")
        print(f"  - tagset_size: {len(tag_to_ix)}")

        # 创建模型实例
        model = cls(
            vocab_size=vocab_size,
            tag_to_ix=tag_to_ix,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers
        )

        # 加载状态字典
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        return model, checkpoint


class BiLSTM_CRF_Compat(nn.Module):
    """兼容版本，使用固定的已知参数"""

    def __init__(self, vocab_size, tag_to_ix, embedding_dim=100, hidden_dim=256):
        """
        使用训练时的确切参数
        """
        super(BiLSTM_CRF_Compat, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # 词嵌入层
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        # LSTM层 - 使用训练时的确切参数
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=2, bidirectional=True, batch_first=True)

        # 输出层
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # CRF层
        self.crf = CRF(self.tagset_size, batch_first=True)

    def forward(self, sentence, tags=None, mask=None):
        embeds = self.word_embeds(sentence)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)


def load_lstm_model_simple(checkpoint_path, device='cpu'):
    """简单加载方法，使用训练时的确切参数"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 使用训练时的确切参数
    vocab_size = len(checkpoint['word_to_ix'])
    tag_to_ix = checkpoint['tag_to_ix']

    # 根据错误信息，训练时使用的是这些参数：
    # embedding_dim=100, hidden_dim=256, lstm_layers=2
    model = BiLSTM_CRF_Compat(
        vocab_size=vocab_size,
        tag_to_ix=tag_to_ix,
        embedding_dim=100,  # 训练时的确切值
        hidden_dim=256  # 训练时的确切值
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint