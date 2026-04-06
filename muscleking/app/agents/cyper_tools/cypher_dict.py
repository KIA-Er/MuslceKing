"""
预定义健身 Cypher 查询字典
基于 fitness_kg 模块的问题类型和 Neo4j 健身知识图谱 schema 设计
对应 fitness_tools_list.py 中 predefined_cypher 工具描述的 8 大类查询
"""

from typing import Dict

predefined_cypher_dict: Dict[str, str] = {
    # ==================== 1. 动作属性查询 (exercise_property) ====================
    "exercise_description": """
MATCH (e:Exercise {name: $exercise_name})
RETURN e.name AS 动作名称, e.description AS 动作说明
""",
    "exercise_difficulty": """
MATCH (e:Exercise {name: $exercise_name})-[:HAS_DIFFICULTY]->(d:Difficulty)
RETURN e.name AS 动作名称, d.name AS 难度
""",
    "exercise_equipment": """
MATCH (e:Exercise {name: $exercise_name})-[:USES_EQUIPMENT]->(eq:Equipment)
RETURN e.name AS 动作名称, collect(eq.name) AS 器械
""",
    "exercise_complete_info": """
MATCH (e:Exercise {name: $exercise_name})
OPTIONAL MATCH (e)-[:TARGETS_MUSCLE]->(m:Muscle)
OPTIONAL MATCH (e)-[:USES_EQUIPMENT]->(eq:Equipment)
OPTIONAL MATCH (e)-[:HAS_DIFFICULTY]->(d:Difficulty)
OPTIONAL MATCH (e)-[:SUPPORTS_GOAL]->(g:TrainingGoal)
RETURN e.name AS 动作名称,
       e.description AS 动作说明,
       collect(DISTINCT m.name) AS 目标肌群,
       collect(DISTINCT eq.name) AS 器械,
       collect(DISTINCT d.name) AS 难度,
       collect(DISTINCT g.name) AS 训练目标
""",
    # ==================== 2. 属性约束查询 (property_constraint) ====================
    "exercises_by_difficulty": """
MATCH (e:Exercise)-[:HAS_DIFFICULTY]->(d:Difficulty {name: $difficulty_name})
RETURN e.name AS 动作名称 LIMIT 15
""",
    "exercises_by_equipment": """
MATCH (e:Exercise)-[:USES_EQUIPMENT]->(eq:Equipment {name: $equipment_name})
RETURN e.name AS 动作名称 LIMIT 15
""",
    "exercises_by_goal": """
MATCH (e:Exercise)-[:SUPPORTS_GOAL]->(g:TrainingGoal {name: $goal_name})
RETURN e.name AS 动作名称 LIMIT 15
""",
    "exercises_by_multi_constraints": """
MATCH (e:Exercise)
WHERE
  EXISTS((e)-[:TARGETS_MUSCLE]->(:Muscle {name: $muscle_name}))
  AND EXISTS((e)-[:USES_EQUIPMENT]->(:Equipment {name: $equipment_name}))
RETURN e.name AS 动作名称 LIMIT 15
""",
    # ==================== 3. 关系约束查询 (relationship_constraint) ====================
    "exercises_by_muscle": """
MATCH (e:Exercise)-[:TARGETS_MUSCLE]->(m:Muscle {name: $muscle_name})
RETURN e.name AS 动作名称 LIMIT 15
""",
    "muscles_of_exercise": """
MATCH (e:Exercise {name: $exercise_name})-[:TARGETS_MUSCLE]->(m:Muscle)
RETURN m.name AS 肌群
""",
    "equipment_of_exercise": """
MATCH (e:Exercise {name: $exercise_name})-[:USES_EQUIPMENT]->(eq:Equipment)
RETURN eq.name AS 器械
""",
    # ==================== 4. 训练计划相关查询 (plan_query) ====================
    "exercises_in_plan": """
MATCH (p:WorkoutPlan {name: $plan_name})-[:INCLUDES_EXERCISE]->(e:Exercise)
RETURN e.name AS 动作名称
""",
    "plans_by_goal": """
MATCH (p:WorkoutPlan)-[:SUPPORTS_GOAL]->(g:TrainingGoal {name: $goal_name})
RETURN p.name AS 训练计划 LIMIT 10
""",
    # ==================== 5. 动作步骤查询 ====================
    "exercise_steps": """
MATCH (e:Exercise {name: $exercise_name})-[:HAS_STEP]->(s:ExerciseStep)
RETURN s.order AS 步骤序号, s.instruction AS 步骤说明
ORDER BY s.order
""",
    "exercise_step_by_order": """
MATCH (e:Exercise {name: $exercise_name})-[:HAS_STEP]->(s:ExerciseStep {order: $step_order})
RETURN s.order AS 步骤序号, s.instruction AS 步骤说明
""",
    # ==================== 6. 风险与收益查询 ====================
    "exercise_risks": """
MATCH (e:Exercise {name: $exercise_name})-[:HAS_RISK]->(r:InjuryRisk)
RETURN e.name AS 动作名称, collect(r.name) AS 伤病风险
""",
    "exercise_benefits": """
MATCH (e:Exercise {name: $exercise_name})-[:HAS_BENEFIT]->(b:Benefit)
RETURN e.name AS 动作名称, collect(b.name) AS 训练收益
""",
    "exercise_health_info": """
MATCH (e:Exercise {name: $exercise_name})
OPTIONAL MATCH (e)-[:HAS_RISK]->(r:InjuryRisk)
OPTIONAL MATCH (e)-[:HAS_BENEFIT]->(b:Benefit)
RETURN e.name AS 动作名称,
       collect(DISTINCT r.name) AS 风险,
       collect(DISTINCT b.name) AS 收益
""",
    # ==================== 7. 统计分析查询 ====================
    "most_targeted_muscles": """
MATCH (e:Exercise)-[:TARGETS_MUSCLE]->(m:Muscle)
WITH m.name AS 肌群, count(e) AS 动作数量
RETURN 肌群, 动作数量
ORDER BY 动作数量 DESC LIMIT 10
""",
    "most_used_equipment": """
MATCH (e:Exercise)-[:USES_EQUIPMENT]->(eq:Equipment)
WITH eq.name AS 器械, count(e) AS 使用次数
RETURN 器械, 使用次数
ORDER BY 使用次数 DESC LIMIT 10
""",
    "exercise_count_by_difficulty": """
MATCH (e:Exercise)-[:HAS_DIFFICULTY]->(d:Difficulty)
WITH d.name AS 难度, count(e) AS 动作数量
RETURN 难度, 动作数量
ORDER BY 动作数量 DESC
""",
    # ==================== 8. 综合推荐查询 ====================
    "recommended_exercises_by_muscle": """
MATCH (e:Exercise)-[:TARGETS_MUSCLE]->(m:Muscle {name: $muscle_name})
RETURN e.name AS 动作名称, e.description AS 动作说明
LIMIT 10
""",
    "similar_exercises": """
MATCH (e1:Exercise {name: $exercise_name})-[:TARGETS_MUSCLE]->(m:Muscle)<-[:TARGETS_MUSCLE]-(e2:Exercise)
WHERE e1 <> e2
WITH e2, count(m) AS 共同肌群数
RETURN e2.name AS 相似动作, 共同肌群数
ORDER BY 共同肌群数 DESC LIMIT 10
""",
}
