"""
预定义健身 Cypher 查询的描述信息.

该模块为 fitness_kg 图谱准备的固定查询提供语义描述,
用于帮助 LLM 根据用户提问快速匹配合适的查询。
描述应覆盖查询意图、适用场景以及可能的自然语言问法提示。
"""

# =========================
# 训练动作属性相关查询
# =========================

EXERCISE_PROPERTY_QUERY_DESCRIPTIONS = {
    "exercise_description": "查询指定训练动作的标准动作描述, 适用于用户想了解动作怎么做。",
    "exercise_target_muscle": "查询某个训练动作主要刺激的目标肌群, 适用于用户关心练哪里。",
    "exercise_equipment": "查询训练动作所需器械或是否徒手, 适用于用户根据器械条件选择动作。",
    "exercise_difficulty": "查询训练动作的难度等级, 适用于新手或进阶用户筛选动作。",
    "exercise_complete_info": "汇总动作的描述、目标肌群、器械与难度等信息, 适用于全面了解某个动作。",
}

# =========================
# 条件筛选类查询
# =========================

FILTER_QUERY_DESCRIPTIONS = {
    "exercises_by_muscle": "根据目标肌群筛选训练动作, 适用于用户想练某个部位。",
    "exercises_by_equipment": "按器械类型筛选训练动作, 适用于用户受限于器械条件。",
    "exercises_by_difficulty": "按难度等级筛选训练动作, 适用于新手或高阶训练需求。",
    "exercises_by_multi_constraints": "按肌群、器械、难度等多条件组合筛选动作, 适用于复杂训练需求。",
}

# =========================
# 动作与肌群关系查询
# =========================

MUSCLE_RELATION_QUERY_DESCRIPTIONS = {
    "muscles_of_exercise": "查询某个训练动作涉及的所有肌群, 适用于用户想了解动作刺激范围。",
    "exercises_for_muscle": "反查某块肌群可以做哪些训练动作, 适用于制定训练计划。",
    "primary_muscle_of_exercise": "查询训练动作的主要发力肌群, 适用于精准训练。",
    "secondary_muscle_of_exercise": "查询训练动作的辅助肌群, 适用于理解协同发力。",
}

# =========================
# 训练计划 / 训练量相关查询
# =========================

WORKOUT_VOLUME_QUERY_DESCRIPTIONS = {
    "recommended_sets_reps": "查询某个动作的推荐组数和次数, 适用于训练安排。",
    "exercise_volume_in_plan": "查询某训练计划中某个动作的训练量, 适用于复盘或调整计划。",
    "total_volume_by_muscle": "统计某块肌群在训练计划中的总训练量, 适用于避免过度或不足训练。",
}

# =========================
# 训练步骤与技巧
# =========================

EXERCISE_STEP_QUERY_DESCRIPTIONS = {
    "exercise_steps": "按顺序列出训练动作的执行步骤, 适用于新手学习动作。",
    "exercise_tips": "查询训练动作的常见要点与注意事项, 适用于避免受伤、提升效果。",
}

# =========================
# 健身知识 / 生理信息
# =========================

FITNESS_INFO_QUERY_DESCRIPTIONS = {
    "muscle_function": "查询肌群的主要功能和作用, 适用于理解训练意义。",
    "training_benefits": "查询某类训练或动作的训练收益, 适用于用户决策是否加入计划。",
    "injury_risk": "查询训练动作可能的受伤风险, 适用于安全训练。",
}

# =========================
# 统计分析类查询
# =========================

STATS_QUERY_DESCRIPTIONS = {
    "most_trained_muscles": "统计训练计划中最常被训练的肌群, 适用于分析训练偏向。",
    "most_used_exercises": "统计最常使用的训练动作, 适用于了解热门动作。",
    "equipment_usage_stats": "统计器械使用频率, 适用于健身房资源分析。",
}

# =========================
# 推荐 / 相似训练查询
# =========================

RECOMMENDATION_QUERY_DESCRIPTIONS = {
    "recommended_exercises_for_muscle": "根据目标肌群推荐合适的训练动作。",
    "similar_exercises": "基于目标肌群或发力模式推荐相似训练动作。",
    "alternative_exercises": "在器械受限或动作不适时推荐替代动作。",
    "exercise_progression": "推荐某个动作的进阶或退阶版本, 适用于能力匹配。",
}

# =========================
# 合并所有查询描述
# =========================

QUERY_DESCRIPTIONS = {}
QUERY_DESCRIPTIONS.update(EXERCISE_PROPERTY_QUERY_DESCRIPTIONS)
QUERY_DESCRIPTIONS.update(FILTER_QUERY_DESCRIPTIONS)
QUERY_DESCRIPTIONS.update(MUSCLE_RELATION_QUERY_DESCRIPTIONS)
QUERY_DESCRIPTIONS.update(WORKOUT_VOLUME_QUERY_DESCRIPTIONS)
QUERY_DESCRIPTIONS.update(EXERCISE_STEP_QUERY_DESCRIPTIONS)
QUERY_DESCRIPTIONS.update(FITNESS_INFO_QUERY_DESCRIPTIONS)
QUERY_DESCRIPTIONS.update(STATS_QUERY_DESCRIPTIONS)
QUERY_DESCRIPTIONS.update(RECOMMENDATION_QUERY_DESCRIPTIONS)
