import requests
import json


def test_demo_functionality():
    """测试演示页面的功能"""
    base_url = "http://localhost:5000"

    print("测试演示页面功能...")

    # 1. 测试健康检查
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✓ 健康检查: {response.status_code}")
        health_data = response.json()
        print(f"  服务状态: {health_data.get('status')}")
        print(f"  可用模型: {health_data.get('models', {}).get('available_models', [])}")
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        return

    # 2. 测试预测接口
    test_text = "李明在北京清华大学读书"
    print(f"\n测试预测接口，文本: {test_text}")

    for model_type in ['bert', 'lstm']:
        try:
            response = requests.post(
                f"{base_url}/ner/predict",
                json={
                    "text": test_text,
                    "model_type": model_type
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                entities = result.get('result', [])
                print(f"✓ {model_type.upper()}模型预测成功")
                print(f"  识别到 {len(entities)} 个实体:")
                for entity in entities:
                    print(f"    - {entity['text']} ({entity['type']})")
            else:
                error_data = response.json()
                print(f"✗ {model_type.upper()}模型预测失败: {response.status_code}")
                print(f"  错误信息: {error_data.get('error', '未知错误')}")

        except requests.exceptions.Timeout:
            print(f"✗ {model_type.upper()}模型请求超时")
        except Exception as e:
            print(f"✗ {model_type.upper()}模型预测异常: {e}")

    # 3. 测试演示页面访问
    try:
        response = requests.get(f"{base_url}/demo")
        print(f"\n✓ 演示页面访问: {response.status_code}")
    except Exception as e:
        print(f"✗ 演示页面访问失败: {e}")


if __name__ == "__main__":
    test_demo_functionality()