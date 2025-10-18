relation_extraction/
├── data/
│   ├── raw_data.json          # 原始标注数据
│   ├── processed_data.json    # 处理后的训练数据
│   └── label_mapping.json     # 标签映射
├── model/
│   ├── train.py              # 模型训练
│   ├── model.py              # 模型定义
│   └── checkpoint/           # 训练好的模型
├── api/
│   ├── app.py               # FastAPI接口
│   └── requirements.txt     # 依赖包
├── config.py               # 配置文件
├── preprocess.py           # 数据预处理
└── README.md              # 项目说明