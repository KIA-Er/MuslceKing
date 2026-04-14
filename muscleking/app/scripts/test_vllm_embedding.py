#!/usr/bin/env python3
"""测试 vLLM Embedding 服务"""

import sys
import requests
import json

def test_health():
    """测试健康检查"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get("http://localhost:50001/health", timeout=5)
        assert response.status_code == 200
        print("✅ Health check passed")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_single_embedding():
    """测试单个文本 embedding"""
    print("\n🔍 Testing single embedding...")
    try:
        payload = {
            "model": "BAAI/bge-m3",
            "input": "如何锻炼胸肌？"
        }
        response = requests.post(
            "http://localhost:50001/v1/embeddings",
            json=payload,
            timeout=30
        )

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert len(data["data"]) == 1
        assert len(data["data"][0]["embedding"]) == 1024

        print(f"✅ Single embedding passed")
        print(f"   - Dimension: {len(data['data'][0]['embedding'])}")
        print(f"   - Sample values: {data['data'][0]['embedding'][:3]}")
        return True
    except Exception as e:
        print(f"❌ Single embedding failed: {e}")
        return False

def test_batch_embedding():
    """测试批量 embedding"""
    print("\n🔍 Testing batch embedding...")
    try:
        payload = {
            "model": "BAAI/bge-m3",
            "input": ["如何锻炼胸肌？", "卧推的正确姿势", "深蹲技巧"]
        }
        response = requests.post(
            "http://localhost:50001/v1/embeddings",
            json=payload,
            timeout=30
        )

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert len(data["data"]) == 3

        for i, item in enumerate(data["data"]):
            assert len(item["embedding"]) == 1024
            print(f"   - Doc {i+1}: dimension={len(item['embedding'])}")

        print("✅ Batch embedding passed")
        return True
    except Exception as e:
        print(f"❌ Batch embedding failed: {e}")
        return False

def main():
    """运行所有测试"""
    print("🧪 vLLM Embedding Service Tests")
    print("=" * 50)

    results = []
    results.append(test_health())
    results.append(test_single_embedding())
    results.append(test_batch_embedding())

    print("\n" + "=" * 50)
    if all(results):
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
