# 模型配置
MODEL_CONFIG = {
    'lstm_crf': {
        'embedding_dim': 100,
        'hidden_dim': 256,
        'lstm_layers': 2
    },
    'bert_lstm_crf': {
        'hidden_dim': 256,
        'lstm_layers': 2,
        'dropout_rate': 0.1
    }
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 8,
    'learning_rate': 2e-5,
    'epochs': 10,
    'max_len': 128
}