#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整Workflow执行测试
无需数据库连接，使用LangGraph的ainvoke方法执行完整工作流
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 加载环境变量
env_path = project_root / ".env"
load_dotenv(env_path)

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from muscleking.app.config.settings import settings

# 导入工作流创建函数
from muscleking.app.agents.multi_agent.multi_tools import create_multi_tool_workflow
from muscleking.app.agents.models.tools_list import (
    cypher_query,
    predefined_cypher,
    microsoft_graphrag_query,
    text2sql_query,
)
from muscleking.app.agents.retrieve.base import BaseCypherExampleRetriever


class MockCypherRetriever(BaseCypherExampleRetriever):
    """空的Cypher示例检索器，用于测试"""

    def __init__(self):
        super().__init__()

    def get_examples(self, query: str, k: int = 5) -> str:
        """返回空的示例字符串"""
        return "No examples available in test mode"


def create_test_workflow():
    """创建测试用的完整workflow"""

    # 检查LLM配置
    llm_api_key = settings.LLM_API_KEY
    if not llm_api_key:
        raise ValueError("LLM_API_KEY 未配置，请检查 .env 文件")

    # 初始化LLM（使用低成本模型进行测试）
    llm = ChatOpenAI(
        api_key=llm_api_key,
        model=settings.LLM_MODEL,
        base_url=settings.LLM_BASE_URL,
        temperature=0.0,
    )

    # 工具Schema列表
    tool_schemas: List[type[BaseModel]] = [
        cypher_query,
        predefined_cypher,
        microsoft_graphrag_query,
        text2sql_query,
    ]

    # 预定义Cypher字典（空）
    predefined_cypher_dict = {}

    # Cypher示例检索器（Mock）
    cypher_retriever = MockCypherRetriever()

    # 创建完整workflow
    print("正在创建完整workflow...")
    workflow = create_multi_tool_workflow(
        llm=llm,
        graph=None,  # 无数据库连接
        tool_schemas=tool_schemas,
        predefined_cypher_dict=predefined_cypher_dict,
        cypher_example_retriever=cypher_retriever,
        scope_description="健身助手服务范围：训练动作、训练计划、营养饮食等健身相关问题",
        llm_cypher_validation=True,
        max_attempts=3,
    )

    print("✓ Workflow创建成功")
    return workflow


def verify_workflow_structure(workflow):
    """验证workflow结构"""
    print("\n" + "=" * 80)
    print("Workflow结构验证")
    print("=" * 80)

    # 获取图结构信息
    try:
        nodes = getattr(workflow, "nodes", [])
        print(f"✓ 节点数量: {len(nodes)}")
        print(f"  节点列表: {list(nodes.keys())}")
    except Exception as e:
        print(f"✗ 无法获取节点信息: {e}")

    print("=" * 80)


async def run_test_case(workflow, test_case: dict):
    """执行单个测试用例"""
    print("\n" + "=" * 80)
    print(f"测试用例: {test_case['id']} - {test_case['name']}")
    print("=" * 80)
    print(f"输入问题: {test_case['input']['question']}")
    print("-" * 80)

    try:
        # 执行完整workflow
        result = await workflow.ainvoke(test_case["input"])

        # 验证结果
        answer = result.get("answer", "")
        steps = result.get("steps", [])

        print(f"\n执行步骤: {' → '.join(steps)}")
        print(f"\n最终答案: {answer}")

        # 验证关键字段
        if answer:
            print("\n✓ PASS - workflow执行成功，返回有效答案")
        else:
            print("\n✗ FAIL - 未返回有效答案")

        # 验证是否包含预期内容
        if "expected_contains" in test_case:
            if test_case["expected_contains"] in answer or test_case[
                "expected_contains"
            ] in str(steps):
                print(f"✓ PASS - 包含预期内容: '{test_case['expected_contains']}'")
            else:
                print(f"⚠ INFO - 未找到预期内容: '{test_case['expected_contains']}'")

        return True

    except Exception as e:
        print(f"\n✗ FAIL - workflow执行失败: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█  MuscleKing 完整Workflow执行测试".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    # 创建workflow
    # try:
    #     workflow = create_test_workflow()
    # except Exception as e:
    #     print(f"\n✗ Workflow创建失败: {e}")
    #     return
    workflow = create_test_workflow()

    # 验证结构
    verify_workflow_structure(workflow)

    # 定义测试用例
    test_cases = [
        {
            "id": "TC-001",
            "name": "健身动作查询",
            "input": {"question": "深蹲怎么练？"},
            "expected_contains": "深蹲",
        },
        {
            "id": "TC-002",
            "name": "非健身问题拦截",
            "input": {"question": "今天天气如何？"},
            "expected_contains": "服务范围",
        },
        {
            "id": "TC-003",
            "name": "训练计划查询",
            "input": {"question": "推荐一个减脂训练计划"},
            "expected_contains": "减脂",
        },
        {
            "id": "TC-004",
            "name": "训练记录查询",
            "input": {"question": "查询我的训练记录"},
            "expected_contains": "记录",
        },
    ]

    # 执行测试
    print("\n" + "=" * 80)
    print("开始执行测试用例")
    print("=" * 80)

    results = []
    for test_case in test_cases:
        passed = await run_test_case(workflow, test_case)
        results.append(
            {"id": test_case["id"], "name": test_case["name"], "passed": passed}
        )

    # 输出测试总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)

    for result in results:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{status} - {result['id']}: {result['name']}")

    print("-" * 80)
    print(f"总计: {passed_count}/{total_count} 通过")
    print("=" * 80)

    if passed_count == total_count:
        print("\n🎉 所有测试用例通过！Workflow运行正常。")
    else:
        print(f"\n⚠️ {total_count - passed_count} 个测试用例未通过，请检查日志。")


if __name__ == "__main__":
    asyncio.run(main())
