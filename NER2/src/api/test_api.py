import requests
import json


def test_api():
    base_url = "http://localhost:5000"

    # 测试健康检查
    print("1. 测试健康检查...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"健康检查: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"健康检查失败: {e}")
        return

    # 测试模型信息
    print("\n2. 测试模型信息...")
    try:
        response = requests.get(f"{base_url}/models/info")
        print(f"模型信息: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"模型信息获取失败: {e}")

    # 测试预测
    print("\n3. 测试预测...")
    test_texts = [
        "李明在北京清华大学读书",
        "马云在杭州阿里巴巴工作",
        "刘德华在香港举办演唱会"
    ]

    for text in test_texts:
        print(f"\n预测文本: {text}")

        # 测试BERT模型
        data = {"text": text, "model_type": "bert"}
        try:
            response = requests.post(f"{base_url}/ner/predict", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"BERT结果: {result['result']}")
            else:
                print(f"BERT预测失败: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"BERT预测异常: {e}")

        # 测试LSTM模型
        data = {"text": text, "model_type": "lstm"}
        try:
            response = requests.post(f"{base_url}/ner/predict", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"LSTM结果: {result['result']}")
            else:
                print(f"LSTM预测失败: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"LSTM预测异常: {e}")


if __name__ == "__main__":
    test_api()