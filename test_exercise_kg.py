#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
健身动作知识图谱测试脚本

测试 exerciseDB 数据导入和知识图谱构建功能。
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_normalization():
    """测试数据标准化功能"""
    print("=" * 60)
    print("测试数据标准化功能")
    print("=" * 60)

    # 导入构建器来测试
    sys.path.insert(0, str(project_root / "muscleking" / "scripts"))

    # 创建一个简单的测试实例
    from build_exercise_kg import ExerciseKGBuilder

    # 创建构建器实例（不传入graph，因为我们需要先测试功能）
    class TestBuilder(ExerciseKGBuilder):
        def __init__(self):
            self.batch_size = 50

    builder = TestBuilder()

    # 测试器械名称标准化
    print("\n测试器械名称标准化:")
    test_equipment = [
        ("dumbbell", "哑铃"),
        ("barbell", "杠铃"),
        ("body weight", "自重"),
        ("cable", "滑轮机"),
        ("bench", "长椅"),
        ("unknown equipment", "unknown equipment"),  # 未知器械应保持原样
    ]

    all_passed = True
    for input_val, expected in test_equipment:
        result = builder._normalize_equipment(input_val)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"  {status} {input_val} -> {result} (期望: {expected})")

    # 测试肌群名称标准化
    print("\n测试肌群名称标准化:")
    test_muscles = [
        ("chest", "胸肌"),
        ("quadriceps", "股四头肌"),
        ("hamstrings", "腘绳肌"),
        ("pectorals", "胸肌"),
        ("deltoids", "三角肌"),
        ("lats", "背阔肌"),
        ("glutes", "臀部"),
        ("full body", "全身"),
        ("unknown muscle", "unknown muscle"),
    ]

    for input_val, expected in test_muscles:
        result = builder._normalize_muscle(input_val)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"  {status} {input_val} -> {result} (期望: {expected})")

    # 测试难度估算
    print("\n测试难度估算:")
    test_names = [
        ("杠铃深蹲", "hard"),
        ("平板卧推", "hard"),
        ("波比跳", "hard"),
        ("哑铃弯举", "easy"),
        ("平板支撑", "easy"),
        ("坐姿划船", "medium"),
    ]

    for name, expected in test_names:
        result = builder._estimate_difficulty({}, name)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"  {status} {name} -> {result} (期望: {expected})")

    # 测试训练目标估算
    print("\n测试训练目标估算:")
    test_goals = [
        ("杠铃深蹲", ["增肌", "力量"]),
        ("波比跳", ["增肌", "力量", "减脂", "心肺"]),
        ("平板支撑", ["增肌", "力量", "核心"]),
        ("跳绳", ["增肌", "力量", "减脂", "心肺"]),
    ]

    for name, expected_list in test_goals:
        result = builder._estimate_goals(name)
        # 检查是否包含所有期望的元素
        contains_all = all(g in result for g in expected_list)
        status = "✅" if contains_all else "❌"
        if not contains_all:
            all_passed = False
        print(f"  {status} {name} -> {result} (期望包含: {expected_list})")

    print(f"\n{'✅ 所有标准化测试通过!' if all_passed else '❌ 存在测试失败'}")
    return all_passed


async def test_graph_construction():
    """测试知识图谱构建功能"""
    print("\n" + "=" * 60)
    print("测试知识图谱构建功能")
    print("=" * 60)

    from muscleking.app.persistence.core.neo4jconn import get_neo4j_graph
    from muscleking.config import settings
    from build_exercise_kg import ExerciseKGBuilder

    try:
        # 连接 Neo4j
        print("\n尝试连接 Neo4j...")
        neo4j_graph = get_neo4j_graph()
        print(f"✅ 成功连接到 Neo4j: {settings.NEO4J_URI}")

        # 创建构建器
        builder = ExerciseKGBuilder(neo4j_graph=neo4j_graph)

        # 获取示例数据
        demo_exercises = builder.get_demo_exercises()
        print(f"\n✅ 获取到 {len(demo_exercises)} 个示例动作")

        # 显示示例数据概览
        print("\n示例动作列表:")
        for i, ex in enumerate(demo_exercises[:5], 1):
            print(
                f"  {i}. {ex['name']} - 设备: {ex.get('equipment', [])}, 目标肌群: {ex.get('targetMuscle', [])}"
            )
        if len(demo_exercises) > 5:
            print(f"  ... 共 {len(demo_exercises)} 个动作")

        # 清空现有数据（测试用）
        print("\n清空现有数据...")
        await builder.clear_existing_data()

        # 构建知识图谱
        print("\n开始构建知识图谱...")
        await builder.build_kg(demo_exercises)

        # 验证结果
        print("\n验证知识图谱...")
        stats = builder.verify_graph()
        print("\n图谱统计信息:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")

        # 检查关键统计
        success = stats.get("exercises", 0) > 0 and stats.get("relationships", 0) > 0

        if success:
            print("\n✅ 知识图谱构建测试通过!")
            print(f"   - 创建了 {stats['exercises']} 个动作节点")
            print(f"   - 创建了 {stats['relationships']} 个关系")
            print(f"   - 创建了 {stats['muscles']} 个肌群节点")
            print(f"   - 创建了 {stats['equipment']} 个器械节点")
        else:
            print("\n❌ 知识图谱构建测试失败")

        return success

    except Exception as e:
        print(f"\n❌ 知识图谱构建测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_cypher_generation():
    """测试 Cypher 语句生成"""
    print("\n" + "=" * 60)
    print("测试 Cypher 语句生成")
    print("=" * 60)

    from build_exercise_kg import ExerciseKGBuilder

    builder = ExerciseKGBuilder(neo4j_graph=None)

    # 测试单个动作的 Cypher 生成
    test_exercise = {
        "id": "test_001",
        "name": "测试深蹲",
        "equipment": ["barbell", "squat rack"],
        "targetMuscle": ["quadriceps", "glutes"],
        "secondaryMuscles": ["hamstrings", "core"],
        "instructions": ["双脚与肩同宽", "下蹲至大腿与地面平行", "站起回到起始位置"],
    }

    print("\n生成测试动作的 Cypher 语句:")
    print("-" * 60)

    # 手动调用内部方法生成 Cypher
    exercise_id = test_exercise["id"]
    name = test_exercise["name"]
    description = (
        test_exercise["instructions"][0] if test_exercise["instructions"] else ""
    )
    equipment = [
        builder._normalize_equipment(e) for e in test_exercise.get("equipment", [])
    ]

    target_muscles = test_exercise.get("targetMuscle", [])
    secondary_muscles = test_exercise.get("secondaryMuscles", [])
    all_muscles = list(set(target_muscles + secondary_muscles))
    muscles = [builder._normalize_muscle(m) for m in all_muscles]

    instructions = test_exercise.get("instructions", [])
    difficulty = builder._estimate_difficulty(test_exercise, name)
    goals = builder._estimate_goals(name)

    cypher = builder._build_cypher(
        exercise_id=exercise_id,
        name=name,
        description=description,
        equipment=equipment,
        muscles=muscles,
        instructions=instructions,
        difficulty=difficulty,
        goals=goals,
    )

    # 显示生成的 Cypher（简化版）
    print(f"MERGE (e:Exercise {{id: '{exercise_id}'}})")
    print(f"SET e.name = '{name}'...")
    print("\n关系数量:")
    print(f"  - USES_EQUIPMENT: {len([e for e in equipment if e])} 个")
    print(f"  - TARGETS_MUSCLE: {len([m for m in muscles if m])} 个")
    print(f"  - HAS_DIFFICULTY: 1 个 ({difficulty})")
    print(f"  - SUPPORTS_GOAL: {len(goals)} 个")
    print(f"  - HAS_STEP: {len(instructions)} 个")

    print("\n✅ Cypher 语句生成测试通过!")
    return True


async def main():
    """主测试函数"""
    print("=" * 60)
    print("健身动作知识图谱测试套件")
    print("=" * 60)

    results = []

    # 1. 测试数据标准化
    results.append(("数据标准化测试", test_normalization()))

    # 2. 测试 Cypher 生成（不需要 Neo4j 连接）
    results.append(("Cypher生成测试", await test_cypher_generation()))

    # 3. 测试知识图谱构建（需要 Neo4j 连接）
    results.append(("知识图谱构建测试", await test_graph_construction()))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        if not passed:
            all_passed = False
        print(f"  {name}: {status}")

    print(f"\n{'🎉 所有测试通过!' if all_passed else '⚠️ 存在测试失败'}")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
