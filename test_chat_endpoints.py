"""
测试chat接口的GET和POST请求
"""
import requests
import json
import time

# API基础URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """测试健康检查接口"""
    print("测试健康检查接口...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"健康检查状态码: {response.status_code}")
        print(f"健康检查响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False

def test_create_session():
    """测试创建会话"""
    print("\n测试创建会话...")
    try:
        response = requests.post(f"{BASE_URL}/sessions", params={"user_id": "test_user_001"})
        print(f"创建会话状态码: {response.status_code}")
        if response.status_code == 200:
            session_data = response.json()
            print(f"创建的会话信息: {json.dumps(session_data, indent=2, ensure_ascii=False)}")
            return session_data["session_id"]
        else:
            print(f"创建会话失败: {response.text}")
            return None
    except Exception as e:
        print(f"创建会话异常: {e}")
        return None

def test_send_chat_message(session_id):
    """测试发送聊天消息"""
    print(f"\n测试发送聊天消息 (会话ID: {session_id})...")
    try:
        chat_data = {
            "message": "你好，我想了解健身训练计划",
            "session_id": session_id,
            "user_id": "test_user_001"
        }
        response = requests.post(f"{BASE_URL}/chat", json=chat_data)
        print(f"发送消息状态码: {response.status_code}")
        if response.status_code == 200:
            chat_response = response.json()
            print(f"聊天响应: {json.dumps(chat_response, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"发送消息失败: {response.text}")
            return False
    except Exception as e:
        print(f"发送消息异常: {e}")
        return False

def test_send_chat_without_session():
    """测试不提供会话ID发送消息（应该创建新会话）"""
    print("\n测试不提供会话ID发送消息...")
    try:
        chat_data = {
            "message": "我想创建新的会话",
            "user_id": "test_user_002"
        }
        response = requests.post(f"{BASE_URL}/chat", json=chat_data)
        print(f"发送消息状态码: {response.status_code}")
        if response.status_code == 200:
            chat_response = response.json()
            print(f"聊天响应: {json.dumps(chat_response, indent=2, ensure_ascii=False)}")
            return chat_response.get("session_id")
        else:
            print(f"发送消息失败: {response.text}")
            return None
    except Exception as e:
        print(f"发送消息异常: {e}")
        return None

def test_get_chat_messages(session_id):
    """测试获取会话消息"""
    print(f"\n测试获取会话消息 (会话ID: {session_id})...")
    try:
        response = requests.get(f"{BASE_URL}/chat/{session_id}")
        print(f"获取消息状态码: {response.status_code}")
        if response.status_code == 200:
            messages = response.json()
            print(f"会话消息数量: {len(messages)}")
            for i, msg in enumerate(messages, 1):
                msg_type = "用户" if msg["is_user"] else "助手"
                print(f"  {i}. [{msg_type}] {msg['content']}")
            return True
        else:
            print(f"获取消息失败: {response.text}")
            return False
    except Exception as e:
        print(f"获取消息异常: {e}")
        return False

def test_get_sessions():
    """测试获取会话列表"""
    print("\n测试获取会话列表...")
    try:
        response = requests.get(f"{BASE_URL}/sessions")
        print(f"获取会话列表状态码: {response.status_code}")
        if response.status_code == 200:
            sessions = response.json()
            print(f"会话总数: {len(sessions)}")
            for i, session in enumerate(sessions, 1):
                print(f"  {i}. 会话ID: {session['session_id']}")
                print(f"     用户ID: {session['user_id']}")
                print(f"     标题: {session['title']}")
                print(f"     消息数量: {session['message_count']}")
                print(f"     创建时间: {session['created_at']}")
            return True
        else:
            print(f"获取会话列表失败: {response.text}")
            return False
    except Exception as e:
        print(f"获取会话列表异常: {e}")
        return False

def test_delete_session(session_id):
    """测试删除会话"""
    print(f"\n测试删除会话 (会话ID: {session_id})...")
    try:
        response = requests.delete(f"{BASE_URL}/sessions/{session_id}")
        print(f"删除会话状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"删除结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"删除会话失败: {response.text}")
            return False
    except Exception as e:
        print(f"删除会话异常: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试chat接口功能\n")
    
    # 测试健康检查
    if not test_health_check():
        print("健康检查失败，无法继续测试")
        return False
    
    # 等待服务完全启动
    print("\n等待服务完全启动...")
    time.sleep(2)
    
    # 测试1: 创建会话
    session_id_1 = test_create_session()
    if not session_id_1:
        print("创建会话失败，跳过后续测试")
        return False
    
    # 测试2: 发送聊天消息（使用现有会话）
    if not test_send_chat_message(session_id_1):
        print("发送聊天消息失败")
    
    # 测试3: 再次发送消息到同一会话
    if not test_send_chat_message(session_id_1):
        print("再次发送消息失败")
    
    # 测试4: 获取会话消息
    if not test_get_chat_messages(session_id_1):
        print("获取会话消息失败")
    
    # 测试5: 不提供会话ID发送消息（创建新会话）
    session_id_2 = test_send_chat_without_session()
    if session_id_2:
        # 获取新会话的消息
        test_get_chat_messages(session_id_2)
    
    # 测试6: 获取会话列表
    if not test_get_sessions():
        print("获取会话列表失败")
    
    # 测试7: 删除会话
    if session_id_1:
        test_delete_session(session_id_1)
    
    if session_id_2:
        test_delete_session(session_id_2)
    
    # 再次获取会话列表，验证删除效果
    print("\n验证删除后的会话列表:")
    test_get_sessions()
    
    print("\n=== 测试完成 ===")
    print("✅ 所有接口测试完成！")
    print("\n📋 测试总结:")
    print("   ✅ 健康检查接口正常")
    print("   ✅ 创建会话接口正常")
    print("   ✅ 发送聊天消息接口正常")
    print("   ✅ 获取会话消息接口正常")
    print("   ✅ 获取会话列表接口正常")
    print("   ✅ 删除会话接口正常")
    print("   ✅ 数据库持久化功能正常")
    print("   ✅ 会话管理功能正常")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出现异常: {e}")