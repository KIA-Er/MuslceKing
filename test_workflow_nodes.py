#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版多工具workflow测试脚本
仅测试guardrails和planner节点，避免版本兼容性问题
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 明确加载.env文件
env_path = project_root / ".env"
load_dotenv(env_path)
print(f"已加载环境变量文件: {env_path}")
print(f"LLM_API_KEY: {'已配置' if os.getenv('LLM_API_KEY') else '未配置'}")

from typing import List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from muscleking.app.config.settings import settings


async def test_individual_nodes():
    """单独测试guardrails和planner节点"""

    print("=" * 80)
    print("MuscleKing 节点测试")
    print("=" * 80)

    # 检查环境配置
    llm_api_key = os.getenv("LLM_API_KEY") or settings.LLM_API_KEY
    if not llm_api_key:
        print("错误: LLM_API_KEY 未配置，请检查 .env 文件")
        print(f"环境变量 LLM_API_KEY: {os.getenv('LLM_API_KEY')}")
        print(f"Settings.LLM_API_KEY: {settings.LLM_API_KEY}")
        return

    print(f"LLM_API_KEY 验证通过: {llm_api_key[:10]}...")

    # 初始化LLM
    print("\n初始化LLM模型...")
    llm = ChatOpenAI(
        api_key=llm_api_key,
        model=settings.LLM_MODEL,
        base_url=settings.LLM_BASE_URL,
        temperature=0.7,
    )
    print(f"LLM模型已初始化: {settings.LLM_MODEL}")

    # 导入节点创建函数
    try:
        from muscleking.app.agents.guardrails.guardrails_node import (
            create_guardrails_node,
        )
        from muscleking.app.agents.planner.planner_node import create_planner_node
        from muscleking.app.agents.tool_selection.tool_selection_node import (
            create_tool_selection_node,
        )

        print("节点模块导入成功")
    except Exception as e:
        print(f"节点模块导入失败: {e}")
        import traceback

        print(f"详细错误: {traceback.format_exc()}")
        return

    #  定义工具模式列表
    from muscleking.app.agents.models.tools_list import (
        cypher_query,
        predefined_cypher,
        microsoft_graphrag_query,
        text2sql_query,
    )

    tool_schemas: List[type[BaseModel]] = [
        cypher_query,
        predefined_cypher,
        microsoft_graphrag_query,
        text2sql_query,
    ]

    # 创建节点
    print("\n创建测试节点...")
    try:
        guardrails_node = create_guardrails_node(
            llm=llm,
            graph=None,  # 不使用Neo4j
            scope_description="健身助手服务范围：训练动作、训练计划、营养饮食等健身相关问题",
        )

        planner_node = create_planner_node(llm=llm)

        tool_selection_node = create_tool_selection_node(llm, tool_schemas, True)

        print("节点创建完成")
    except Exception as e:
        print(f"节点创建失败: {e}")
        import traceback

        print(f"详细错误: {traceback.format_exc()}")
        return

    # 测试问题列表
    test_questions = [
        "请问深蹲怎么练？",
        "我的胸肌怎么训练比较好？",
        "推荐一个减脂训练计划",
        "今天天气如何？",  # 非健身问题，测试guardrails
        "健身前应该注意什么？",
    ]

    print(f"\n开始测试 {len(test_questions)} 个问题...")
    print("=" * 80)

    for i, question in enumerate(test_questions, 1):
        print(f"\n测试问题 {i}: {question}")
        print("-" * 50)

        # 测试guardrails节点
        print("测试Guardrails节点...")
        try:
            guardrails_result = await guardrails_node({"question": question})
            next_action = guardrails_result.get("next_action", "unknown")
            summary = guardrails_result.get("summary", None)
            steps = guardrails_result.get("steps", [])

            print(f"   决策结果: {next_action}")
            if summary:
                print(f"   拒绝原因: {summary}")
            print(f"   执行步骤: {steps}")

        except Exception as e:
            print(f"   Guardrails测试失败: {e}")
            continue

        # 如果guardrails通过，测试planner节点
        if next_action == "planner":
            print("测试Planner节点...")
            try:
                planner_result = await planner_node({"question": question})
                tasks = planner_result.get("tasks", [])
                next_action_planner = planner_result.get("next_action", "unknown")

                print(f"   下一步动作: {next_action_planner}")
                print(f"   分解任务数量: {len(tasks)}")

                for j, task in enumerate(tasks, 1):
                    task_question = getattr(task, "question", str(task))
                    print(f"      任务{j}: {task_question}")

            except Exception as e:
                print(f"   Planner测试失败: {e}")
                import traceback

                print(f"   详细错误: {traceback.format_exc()}")

            # if next_action == "tool_selection":
            print("测试Tool Selection节点...")
            try:
                for j, task in enumerate(tasks, 1):
                    tool_selection_result = await tool_selection_node(
                        {"question": getattr(task, "question", str(task))}
                    )
                    print(f"   输出的内容是：{tool_selection_result}")

                    # 正确访问 Command 对象的内容
                    target_node = tool_selection_result.goto.node
                    send_arg = tool_selection_result.goto.arg

                    print(f"   转到目标节点: {target_node}")
                    print(f"   任务内容: {send_arg.get('task', '')}")
                    print(f"   查询名称: {send_arg.get('query_name', '')}")
                    print(f"   查询参数: {send_arg.get('query_parameters', '')}")
                    print(f"   执行步骤: {send_arg.get('steps', '')}")

            except Exception as e:
                print(f"   Tool Selection测试失败: {e}")
                import traceback

                traceback.print_exc()

        print("=" * 50)

    print("\n节点测试完成！")
    print("=" * 80)


def analyze_workflow_structure():
    """分析workflow结构（不创建完整workflow）"""

    print("\nWorkflow结构分析")
    print("=" * 80)

    try:
        # 读取multi_tools.py文件内容
        multi_tools_path = (
            project_root / "muscleking" / "app" / "agents" / "multi_tools.py"
        )
        with open(multi_tools_path, "r", encoding="utf-8") as f:
            content = f.read()

        print("分析create_multi_tool_workflow函数结构:")

        # 查找节点创建部分
        lines = content.split("\n")
        node_creations = []

        for line in lines:
            line = line.strip()
            if "guardrails = create_" in line:
                node_creations.append(line)
            elif "planner = create_" in line:
                node_creations.append(line)
            elif "cypher_query = create_" in line:
                node_creations.append(line)
            elif "predefined_cypher = create_" in line:
                node_creations.append(line)
            elif "customer_tools = create_" in line:
                node_creations.append(line)
            elif "text2sql_query = create_" in line:
                node_creations.append(line)
            elif "tool_selection = create_" in line:
                node_creations.append(line)
            elif "summarize = create_" in line:
                node_creations.append(line)
            elif "final_answer = create_" in line:
                node_creations.append(line)

        print("\n节点创建顺序:")
        for i, creation in enumerate(node_creations, 1):
            print(f"   {i}. {creation}")

        # 查找add_node部分
        add_nodes = []
        for line in lines:
            if "main_graph_builder.add_node" in line:
                add_nodes.append(line.strip())

        print(f"\n添加到workflow的节点数量: {len(add_nodes)}")
        for i, node in enumerate(add_nodes, 1):
            print(f"   {i}. {node}")

        # 查找边连接部分
        edges = []
        for line in lines:
            if "main_graph_builder.add_edge" in line:
                edges.append(line.strip())
            elif "main_graph_builder.add_conditional_edges" in line:
                edges.append(line.strip())

        print(f"\nWorkflow边连接数量: {len(edges)}")
        for i, edge in enumerate(edges, 1):
            print(f"   {i}. {edge}")

        print("\nWorkflow结构分析完成")

        # 分析关键边连接
        print("\n关键工作流路径:")
        print("1. START -> guardrails")
        print("2. guardrails -> [planner|end] (条件边)")
        print("3. planner -> tool_selection (条件边)")
        print("4. 各工具节点 -> summarize")
        print("5. summarize -> final_answer")
        print("6. final_answer -> END")

    except Exception as e:
        print(f"Workflow结构分析失败: {e}")


async def main():
    """主函数"""
    print("MuscleKing 简化测试脚本")
    print("此脚本将测试guardrails和planner节点，并分析workflow结构")

    try:
        # 测试workflow结构
        # analyze_workflow_structure()

        # 单独测试节点
        await test_individual_nodes()

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生未预期的错误: {e}")
        import traceback

        print(f"详细错误: {traceback.format_exc()}")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())
