from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizerFast
import os
import sys
import json

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 现在导入模型类
from models.bert_lstm_crf_model import BertBiLSTM_CRF
from models.lstm_crf_model import load_lstm_model_simple

app = Flask(__name__)

# 配置
LOCAL_BERT_PATH = "G:\\Pycharm\\Project\\dataroot\\models\\bert-base-chinese"
MODEL_BASE_PATH = "G:\\Pycharm\\Project\\AI_learning\\20231118\\NLPProject05\\NER2\\src\\train\\models"


def find_model_files():
    """查找模型文件"""
    model_files = {}

    # 你的模型文件路径
    bert_model_path = os.path.join(MODEL_BASE_PATH, "bert_lstm_crf_model.pth")
    lstm_model_path = os.path.join(MODEL_BASE_PATH, "lstm_crf_model.pth")

    if os.path.exists(bert_model_path):
        model_files['bert'] = bert_model_path
        print(f"✓ 找到BERT模型: {bert_model_path}")
    else:
        print(f"✗ BERT模型不存在: {bert_model_path}")

    if os.path.exists(lstm_model_path):
        model_files['lstm'] = lstm_model_path
        print(f"✓ 找到LSTM模型: {lstm_model_path}")
    else:
        print(f"✗ LSTM模型不存在: {lstm_model_path}")

    return model_files


def load_models():
    """加载所有模型"""
    models = {}
    model_files = find_model_files()

    print("开始加载模型...")

    # 加载LSTM+CRF模型 - 使用简单加载方法
    if 'lstm' in model_files:
        try:
            lstm_path = model_files['lstm']
            print(f"加载LSTM模型: {lstm_path}")

            # 使用简单加载方法
            lstm_crf_model, lstm_checkpoint = load_lstm_model_simple(lstm_path)

            models['lstm_crf'] = {
                'model': lstm_crf_model,
                'checkpoint': lstm_checkpoint
            }
            print("✓ LSTM+CRF模型加载成功")
            print(f"  词汇表大小: {len(lstm_checkpoint['word_to_ix'])}")
            print(f"  标签数量: {len(lstm_checkpoint['tag_to_ix'])}")

        except Exception as e:
            print(f"✗ 加载LSTM+CRF模型失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ LSTM模型文件不存在")

    # 加载Bert+LSTM+CRF模型
    if 'bert' in model_files:
        try:
            bert_path = model_files['bert']
            print(f"加载BERT模型: {bert_path}")
            bert_checkpoint = torch.load(bert_path, map_location='cpu')

            print(f"BERT模型检查点键: {list(bert_checkpoint.keys())}")

            # 使用保存的BERT路径或默认路径
            bert_model_path = bert_checkpoint.get('bert_path', LOCAL_BERT_PATH)
            print(f"使用BERT路径: {bert_model_path}")

            bert_model = BertBiLSTM_CRF(bert_model_path, bert_checkpoint['tag_to_ix'])
            bert_model.load_state_dict(bert_checkpoint['model_state_dict'])
            bert_model.eval()

            # 加载快速分词器
            if os.path.exists(bert_model_path):
                print(f"从本地加载tokenizer: {bert_model_path}")
                bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
            else:
                print("从网络加载tokenizer")
                bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

            models['bert_crf'] = {
                'model': bert_model,
                'tokenizer': bert_tokenizer,
                'checkpoint': bert_checkpoint
            }
            print("✓ Bert+LSTM+CRF模型加载成功")
            print(f"  标签数量: {len(bert_checkpoint['tag_to_ix'])}")

        except Exception as e:
            print(f"✗ 加载Bert+LSTM+CRF模型失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ BERT模型文件不存在")

    return models


# 全局变量存储模型
print("初始化模型...")
models = load_models()


@app.route('/ner/predict', methods=['POST'])
def predict_ner():
    """
    地址实体识别预测接口
    POST数据格式:
    {
        "text": "要识别的文本",
        "model_type": "bert" 或 "lstm"
    }
    """
    data = request.json
    text = data.get('text', '')
    model_type = data.get('model_type', 'bert')  # 'lstm' or 'bert'

    if not text:
        return jsonify({'error': '文本不能为空'}), 400

    if model_type == 'lstm' and 'lstm_crf' in models:
        result = predict_lstm_crf(text)
    elif model_type == 'bert' and 'bert_crf' in models:
        result = predict_bert_crf(text)
    else:
        available_models = list(models.keys())
        return jsonify({
            'error': f'模型类型不支持或模型未加载',
            'available_models': available_models,
            'requested_model': f'{model_type}_crf',
            'help': '可用的模型类型: ' + ', '.join([m.replace('_crf', '') for m in available_models])
        }), 400

    return jsonify({'result': result})


def predict_lstm_crf(text):
    """使用LSTM+CRF模型预测"""
    model_info = models['lstm_crf']
    model = model_info['model']
    checkpoint = model_info['checkpoint']

    words = list(text)
    word_to_ix = checkpoint['word_to_ix']
    ix_to_tag = checkpoint['ix_to_tag']

    print(f"LSTM预测: {text}")
    print(f"词汇表大小: {len(word_to_ix)}")

    # 转换为索引，未知词使用0
    sentence_in = torch.tensor([word_to_ix.get(w, 0) for w in words]).unsqueeze(0)
    mask = torch.ones_like(sentence_in).bool()

    with torch.no_grad():
        tags = model(sentence_in, mask=mask)

    entities = extract_entities(words, tags[0], ix_to_tag)
    print(f"LSTM预测结果: {entities}")
    return entities


def predict_bert_crf(text):
    """使用Bert+CRF模型预测"""
    model_info = models['bert_crf']
    model = model_info['model']
    tokenizer = model_info['tokenizer']
    checkpoint = model_info['checkpoint']

    words = list(text)
    ix_to_tag = {v: k for k, v in checkpoint['tag_to_ix'].items()}

    print(f"BERT预测: {text}")
    print(f"标签映射: {ix_to_tag}")

    # 使用快速分词器对输入进行编码
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt',
        return_offsets_mapping=True
    )

    with torch.no_grad():
        tags = model(encoding['input_ids'], encoding['attention_mask'])

    # 处理预测结果 - 使用快速分词器的word_ids方法
    word_ids = encoding.word_ids(batch_index=0)
    predicted_tags = []

    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and i < len(tags[0]):
            predicted_tags.append(ix_to_tag.get(tags[0][i], 'O'))

    # 只取实际文本长度的标签
    predicted_tags = predicted_tags[:len(words)]

    entities = extract_entities(words, predicted_tags, ix_to_tag)
    print(f"BERT预测结果: {entities}")
    return entities


def extract_entities(words, tags, ix_to_tag):
    """从标签序列中提取实体 - 支持B/I/E标记"""
    entities = []
    current_entity = None
    entity_type = None

    for i, (word, tag_idx) in enumerate(zip(words, tags)):
        if isinstance(tag_idx, int):
            tag = ix_to_tag.get(tag_idx, 'O')
        else:
            tag = tag_idx

        # 处理B-开头的标签（实体开始）
        if tag.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            entity_type = tag[2:]  # 去掉B-前缀
            current_entity = {
                'text': word,
                'type': entity_type,
                'start': i,
                'end': i
            }
        # 处理I-或E-开头的标签（实体中间或结束）
        elif (tag.startswith('I-') or tag.startswith('E-')) and current_entity and entity_type == tag[2:]:
            current_entity['text'] += word
            current_entity['end'] = i
            # 如果是E-标签，表示实体结束
            if tag.startswith('E-'):
                entities.append(current_entity)
                current_entity = None
                entity_type = None
        else:
            # 其他情况（O标签或实体类型不匹配）
            if current_entity:
                entities.append(current_entity)
                current_entity = None
                entity_type = None

    # 处理最后一个实体
    if current_entity:
        entities.append(current_entity)

    return entities


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    model_status = {
        'lstm_crf_loaded': 'lstm_crf' in models,
        'bert_crf_loaded': 'bert_crf' in models,
        'available_models': list(models.keys())
    }
    return jsonify({'status': 'ok', 'models': model_status})


@app.route('/models/info', methods=['GET'])
def models_info():
    """获取模型详细信息"""
    info = {}

    if 'lstm_crf' in models:
        checkpoint = models['lstm_crf']['checkpoint']
        model = models['lstm_crf']['model']
        info['lstm_crf'] = {
            'vocab_size': len(checkpoint['word_to_ix']),
            'tag_size': len(checkpoint['tag_to_ix']),
            'tags': list(checkpoint['tag_to_ix'].keys()),
            'embedding_dim': model.embedding_dim,
            'hidden_dim': model.hidden_dim
        }

    if 'bert_crf' in models:
        checkpoint = models['bert_crf']['checkpoint']
        info['bert_crf'] = {
            'tag_size': len(checkpoint['tag_to_ix']),
            'tags': list(checkpoint['tag_to_ix'].keys())
        }

    return jsonify(info)


@app.route('/demo', methods=['GET'])
def demo():
    """演示页面"""
    demo_html = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>地址实体识别演示</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .header p {
                opacity: 0.9;
                font-size: 1.1em;
            }
            .content {
                padding: 40px;
            }
            .input-group {
                margin-bottom: 30px;
            }
            .input-group label {
                display: block;
                margin-bottom: 10px;
                font-weight: 600;
                color: #2c3e50;
                font-size: 1.1em;
            }
            textarea {
                width: 100%;
                height: 120px;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 16px;
                resize: vertical;
                transition: border-color 0.3s;
                font-family: inherit;
            }
            textarea:focus {
                outline: none;
                border-color: #3498db;
            }
            .model-selection {
                display: flex;
                gap: 20px;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }
            .model-option {
                flex: 1;
                min-width: 200px;
            }
            .model-option label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #2c3e50;
            }
            .radio-group {
                display: flex;
                gap: 15px;
            }
            .radio-item {
                display: flex;
                align-items: center;
                gap: 8px;
                cursor: pointer;
            }
            .radio-item input {
                width: 18px;
                height: 18px;
            }
            .btn {
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                width: 100%;
                font-family: inherit;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(52, 152, 219, 0.3);
            }
            .btn:active {
                transform: translateY(0);
            }
            .btn:disabled {
                background: #bdc3c7;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            .result {
                margin-top: 30px;
                padding: 25px;
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 5px solid #3498db;
            }
            .result h3 {
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 1.3em;
            }
            .entities {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 20px;
            }
            .entity {
                padding: 8px 15px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 14px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                color: white;
            }
            /* 地址实体类型颜色 */
            .entity.prov { background: #ff6b6b; }
            .entity.city { background: #4ecdc4; }
            .entity.district { background: #45b7d1; }
            .entity.community { background: #96ceb4; }
            .entity.town { background: #feca57; }
            .entity.poi { background: #ff9ff3; }
            .entity.road { background: #54a0ff; }
            .entity.roadno { background: #5f27cd; }
            .entity.subpoi { background: #00d2d3; }
            .entity.devzone { background: #ff9f43; }
            .entity.houseno { background: #ee5a24; }
            .entity.intersection { background: #c44569; }
            .highlighted-text {
                background: white;
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                line-height: 1.6;
                font-size: 16px;
            }
            .loading {
                text-align: center;
                padding: 20px;
                color: #666;
            }
            .loading-spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 2s linear infinite;
                margin: 0 auto 15px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error {
                background: #ffeaa7;
                color: #d63031;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                border-left: 5px solid #fab1a0;
            }
            .stats {
                display: flex;
                gap: 20px;
                margin-top: 15px;
                flex-wrap: wrap;
            }
            .stat {
                background: white;
                padding: 10px 15px;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                text-align: center;
                flex: 1;
                min-width: 120px;
            }
            .stat-number {
                font-size: 1.5em;
                font-weight: bold;
                color: #3498db;
            }
            .stat-label {
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }
            .legend {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 20px 0;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
            }
            .legend-item {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 12px;
            }
            .legend-color {
                width: 15px;
                height: 15px;
                border-radius: 3px;
            }
            @media (max-width: 768px) {
                .content {
                    padding: 20px;
                }
                .model-selection {
                    flex-direction: column;
                }
                .header h1 {
                    font-size: 2em;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>地址实体识别系统</h1>
                <p>基于深度学习的地址要素识别演示 - 支持37种地址实体类型</p>
            </div>

            <div class="content">
                <div class="input-group">
                    <label for="textInput">请输入要识别的地址文本：</label>
                    <textarea id="textInput" placeholder="例如：北京市海淀区中关村大街27号北京大学图书馆...">北京市海淀区中关村大街27号北京大学图书馆</textarea>
                </div>

                <div class="model-selection">
                    <div class="model-option">
                        <label>选择识别模型：</label>
                        <div class="radio-group">
                            <div class="radio-item">
                                <input type="radio" id="model-bert" name="model" value="bert" checked>
                                <label for="model-bert">BERT模型</label>
                            </div>
                            <div class="radio-item">
                                <input type="radio" id="model-lstm" name="model" value="lstm">
                                <label for="model-lstm">LSTM模型</label>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 图例 -->
                <div class="legend">
                    <div class="legend-item"><div class="legend-color" style="background: #ff6b6b;"></div>省份</div>
                    <div class="legend-item"><div class="legend-color" style="background: #4ecdc4;"></div>城市</div>
                    <div class="legend-item"><div class="legend-color" style="background: #45b7d1;"></div>区县</div>
                    <div class="legend-item"><div class="legend-color" style="background: #96ceb4;"></div>社区</div>
                    <div class="legend-item"><div class="legend-color" style="background: #feca57;"></div>乡镇</div>
                    <div class="legend-item"><div class="legend-color" style="background: #ff9ff3;"></div>兴趣点</div>
                    <div class="legend-item"><div class="legend-color" style="background: #54a0ff;"></div>道路</div>
                    <div class="legend-item"><div class="legend-color" style="background: #5f27cd;"></div>门牌号</div>
                    <div class="legend-item"><div class="legend-color" style="background: #00d2d3;"></div>子兴趣点</div>
                    <div class="legend-item"><div class="legend-color" style="background: #ff9f43;"></div>开发区</div>
                    <div class="legend-item"><div class="legend-color" style="background: #ee5a24;"></div>房屋编号</div>
                    <div class="legend-item"><div class="legend-color" style="background: #c44569;"></div>交叉路口</div>
                </div>

                <button class="btn" id="predictBtn" onclick="predict()">开始识别地址实体</button>

                <div id="result"></div>
            </div>
        </div>

        <script>
            // 添加控制台日志以便调试
            console.log('页面加载完成');

            function predict() {
                console.log('点击识别按钮');

                const text = document.getElementById('textInput').value.trim();
                const modelType = document.querySelector('input[name="model"]:checked').value;
                const predictBtn = document.getElementById('predictBtn');

                console.log('输入文本:', text);
                console.log('选择模型:', modelType);

                if (!text) {
                    alert('请输入要识别的地址文本！');
                    return;
                }

                // 禁用按钮，防止重复点击
                predictBtn.disabled = true;
                predictBtn.textContent = '识别中...';

                // 显示加载中
                document.getElementById('result').innerHTML = `
                    <div class="loading">
                        <div class="loading-spinner"></div>
                        <p>正在识别地址实体中，请稍候...</p>
                    </div>
                `;

                // 发送请求
                fetch('/ner/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        model_type: modelType
                    })
                })
                .then(response => {
                    console.log('响应状态:', response.status);
                    if (!response.ok) {
                        return response.json().then(err => { throw new Error(err.error || `HTTP错误: ${response.status}`); });
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('响应数据:', data);
                    displayResult(data.result, text, modelType);
                })
                .catch(error => {
                    console.error('请求错误:', error);
                    document.getElementById('result').innerHTML = `
                        <div class="error">
                            <strong>错误：</strong>${error.message}
                            <br><br>
                            <small>请检查：<br>
                            1. 服务是否正常运行<br>
                            2. 网络连接是否正常<br>
                            3. 输入文本格式是否正确</small>
                        </div>
                    `;
                })
                .finally(() => {
                    // 重新启用按钮
                    predictBtn.disabled = false;
                    predictBtn.textContent = '开始识别地址实体';
                });
            }

            function displayResult(entities, originalText, modelType) {
                console.log('显示结果，实体数量:', entities.length);

                let resultHtml = '<div class="result">';

                if (!entities || entities.length === 0) {
                    resultHtml += `
                        <h3>识别结果</h3>
                        <p>未识别到任何地址实体。</p>
                        <p><small>提示：请尝试输入包含地址信息的文本，如"北京市海淀区中关村大街27号"</small></p>
                    `;
                } else {
                    // 统计实体类型
                    const entityCounts = countEntityTypes(entities);

                    resultHtml += `
                        <h3>识别结果（使用${modelType.toUpperCase()}模型）</h3>
                        <div class="stats">
                            <div class="stat">
                                <div class="stat-number">${entities.length}</div>
                                <div class="stat-label">识别到实体数</div>
                            </div>
                            ${Object.entries(entityCounts).map(([type, count]) => `
                                <div class="stat">
                                    <div class="stat-number">${count}</div>
                                    <div class="stat-label">${getEntityTypeName(type)}</div>
                                </div>
                            `).join('')}
                        </div>

                        <h4>识别的地址实体：</h4>
                        <div class="entities">
                    `;

                    entities.forEach((entity, index) => {
                        console.log(`实体 ${index}:`, entity);
                        resultHtml += `<div class="entity ${entity.type}">${entity.text}（${getEntityTypeName(entity.type)}）</div>`;
                    });

                    resultHtml += `</div>`;

                    // 高亮显示文本中的实体
                    let highlightedText = originalText;
                    // 按起始位置排序，避免替换顺序问题
                    const sortedEntities = [...entities].sort((a, b) => b.start - a.start);

                    sortedEntities.forEach(entity => {
                        const entityText = entity.text;
                        const start = entity.start;
                        const end = entity.end !== undefined ? entity.end : start + entityText.length - 1;

                        // 使用位置信息进行精确替换
                        const before = highlightedText.substring(0, start);
                        const middle = `<span class="entity ${entity.type}">${entityText}</span>`;
                        const after = highlightedText.substring(end + 1);
                        highlightedText = before + middle + after;
                    });

                    resultHtml += `
                        <h4>高亮地址文本：</h4>
                        <div class="highlighted-text">${highlightedText}</div>
                    `;
                }

                resultHtml += '</div>';
                document.getElementById('result').innerHTML = resultHtml;
            }

            function countEntityTypes(entities) {
                const counts = {};
                entities.forEach(entity => {
                    counts[entity.type] = (counts[entity.type] || 0) + 1;
                });
                return counts;
            }

            function getEntityTypeName(type) {
                const typeNames = {
                    'prov': '省份',
                    'city': '城市', 
                    'district': '区县',
                    'community': '社区',
                    'town': '乡镇',
                    'poi': '兴趣点',
                    'road': '道路',
                    'roadno': '门牌号',
                    'subpoi': '子兴趣点',
                    'devzone': '开发区',
                    'houseno': '房屋编号',
                    'intersection': '交叉路口'
                };
                return typeNames[type] || type;
            }

            // 添加回车键支持
            document.getElementById('textInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && e.ctrlKey) {
                    predict();
                }
            });

            // 页面加载完成后检查服务状态
            window.addEventListener('load', async () => {
                console.log('检查服务状态...');
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    console.log('服务状态:', data);

                    if (data.status === 'ok') {
                        console.log('服务状态正常，可用模型:', data.models.available_models);
                    }
                } catch (error) {
                    console.error('服务状态检查失败:', error);
                }
            });
        </script>
    </body>
    </html>
    """
    return demo_html


@app.route('/', methods=['GET'])
def index():
    """首页重定向到演示页面"""
    return """
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/demo">
        </head>
        <body>
            <p>正在跳转到演示页面... <a href="/demo">点击这里</a></p>
        </body>
    </html>
    """


@app.route('/api/docs', methods=['GET'])
def api_docs():
    """API文档页面"""
    docs = {
        "API文档": {
            "基本信息": {
                "基础URL": "http://localhost:5000",
                "内容类型": "application/json"
            },
            "接口列表": {
                "健康检查": {
                    "URL": "/health",
                    "方法": "GET",
                    "描述": "检查服务状态和模型加载情况"
                },
                "模型信息": {
                    "URL": "/models/info",
                    "方法": "GET",
                    "描述": "获取加载模型的详细信息"
                },
                "实体识别": {
                    "URL": "/ner/predict",
                    "方法": "POST",
                    "描述": "进行地址实体识别",
                    "请求参数": {
                        "text": "要识别的文本（字符串）",
                        "model_type": "模型类型：'bert' 或 'lstm'"
                    },
                    "响应示例": {
                        "result": [
                            {
                                "text": "实体文本",
                                "type": "实体类型（prov/city/district等）",
                                "start": "起始位置",
                                "end": "结束位置"
                            }
                        ]
                    }
                }
            },
            "实体类型说明": {
                "行政区划": ["prov(省)", "city(市)", "district(区县)", "community(社区)", "town(乡镇)"],
                "地址要素": ["poi(兴趣点)", "road(道路)", "roadno(门牌号)", "subpoi(子兴趣点)", "devzone(开发区)",
                             "houseno(房屋编号)", "intersection(交叉路口)"],
                "标记说明": ["B-(开始)", "I-(中间)", "E-(结束)", "O(非实体)"]
            },
            "使用示例": {
                "curl示例": "curl -X POST http://localhost:5000/ner/predict -H \"Content-Type: application/json\" -d '{\"text\": \"北京市海淀区中关村大街27号\", \"model_type\": \"bert\"}'",
                "Python示例": """
import requests

response = requests.post(
    "http://localhost:5000/ner/predict",
    json={
        "text": "北京市海淀区中关村大街27号",
        "model_type": "bert"
    }
)
print(response.json())
                """
            }
        }
    }
    return jsonify(docs)


@app.route('/debug/lstm', methods=['POST'])
def debug_lstm():
    """调试LSTM模型预测"""
    data = request.json
    text = data.get('text', '北京市海淀区')

    if 'lstm_crf' not in models:
        return jsonify({'error': 'LSTM模型未加载'})

    model_info = models['lstm_crf']
    model = model_info['model']
    checkpoint = model_info['checkpoint']

    words = list(text)
    word_to_ix = checkpoint['word_to_ix']
    ix_to_tag = checkpoint['ix_to_tag']

    # 详细日志
    print(f"调试LSTM预测: {text}")
    print(f"词汇表: {len(word_to_ix)} 个词")
    print(f"输入词汇: {words}")

    # 检查词汇表覆盖
    unknown_words = [w for w in words if w not in word_to_ix]
    if unknown_words:
        print(f"未知词汇: {unknown_words}")

    # 准备输入
    sentence_in = torch.tensor([word_to_ix.get(w, 0) for w in words]).unsqueeze(0)
    mask = torch.ones_like(sentence_in).bool()

    print(f"输入张量: {sentence_in}")
    print(f"掩码: {mask}")

    with torch.no_grad():
        tags = model(sentence_in, mask=mask)

    print(f"原始预测: {tags}")

    # 转换标签
    predicted_tags = [ix_to_tag.get(tag_idx, 'O') for tag_idx in tags[0][:len(words)]]
    print(f"转换后标签: {predicted_tags}")

    # 提取实体
    entities = extract_entities(words, predicted_tags, ix_to_tag)

    return jsonify({
        'text': text,
        'words': words,
        'predicted_tags': predicted_tags,
        'entities': entities,
        'vocab_size': len(word_to_ix),
        'unknown_words': unknown_words
    })


if __name__ == '__main__':
    print("=" * 60)
    print("启动地址实体识别服务...")
    print("可用模型:", list(models.keys()))
    print("模型基础路径:", MODEL_BASE_PATH)
    print("=" * 60)
    print("访问以下地址：")
    print("  http://localhost:5000/demo    - 演示页面")
    print("  http://localhost:5000/health  - 服务状态")
    print("  http://localhost:5000/api/docs - API文档")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)