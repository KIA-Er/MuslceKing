"""
健身知识图谱工具定义
基于 fitness_kg 模块的问题分类和 Neo4j 健身知识图谱模型
用于提示模型每个工具的作用，让模型选择工具
"""

from typing import Optional
from pydantic import BaseModel, Field


class cypher_query(BaseModel):
    """健身知识图谱 Cypher 查询工具

    当用户询问关于训练动作、训练计划、肌群、器械、消耗、营养等信息时，
    使用此工具生成 Cypher 查询语句。

    适用场景包括：
    - 动作属性查询（目标肌群、难度、器械、训练类型等）
    - 训练计划查询（训练频率、时长、目标）
    - 动作执行参数（组数、次数、重量、时间）
    - 基于条件的动作筛选（如：无器械动作、胸部训练）
    - 基于目标的训练推荐（如：减脂训练、增肌训练）
    """

    task: str = Field(
        ..., description="健身相关的查询任务描述，LLM 会根据此任务生成 Cypher 查询语句"
    )


class predefined_cypher(BaseModel):
    """预定义健身 Cypher 查询工具

    基于 fitness_kg 模块的问题类型和 Neo4j 健身知识图谱模型设计，
    用于高频健身查询的快速响应。
    """

    query: str = Field(
        ..., description="预定义查询标识符，对应 fitness_cypher_dict 中的键"
    )

    parameters: dict = Field(
        ...,
        description="查询所需参数，如 {'exercise_name': '深蹲', 'muscle_name': '股四头肌'}",
    )


class fitness_predefined_query_types:
    """
    预定义查询分类说明（文档用途）
    """

    """
    1. 动作属性查询 (exercise_property)
       - exercise_muscles: 查询动作主要/次要刺激肌群
       - exercise_equipment: 查询动作所需器械
       - exercise_difficulty: 查询动作难度
       - exercise_type: 查询动作类型（力量、有氧、拉伸）
       - exercise_complete_info: 查询动作完整信息

    2. 训练计划查询 (workout_plan)
       - plan_overview: 查询训练计划简介
       - plan_duration: 查询计划周期
       - plan_goal: 查询计划目标（增肌/减脂/塑形）
       - exercises_in_plan: 查询计划包含的所有动作

    3. 条件筛选查询 (property_constraint)
       - exercises_by_muscle: 查询某肌群的所有动作
       - exercises_by_equipment: 查询某器械可做的动作
       - exercises_by_goal: 查询符合某训练目标的动作
       - exercises_by_multi_constraints: 多条件组合查询

    4. 动作参数查询 (exercise_volume)
       - sets_and_reps: 查询动作推荐组数和次数
       - weight_recommendation: 查询推荐负重
       - duration_recommendation: 查询推荐训练时长

    5. 训练步骤查询
       - exercise_steps: 查询动作标准执行步骤
       - step_by_order: 查询动作的指定步骤

    6. 营养与消耗查询
       - calories_burned: 查询动作或训练消耗热量
       - nutrition_info: 查询营养素信息
       - nutrition_benefits: 查询营养健康功效
       - nutrition_complete_info: 查询营养完整信息

    7. 统计分析查询
       - most_trained_muscles: 最常训练的肌群
       - popular_exercises: 最常见动作
       - equipment_usage_count: 器械使用统计
       - exercise_count_by_type: 各类型动作数量

    8. 综合推荐与推理
       - workout_recommendation: 基于目标生成训练方案
       - similar_exercises: 查询相似动作
       - alternative_exercises: 查询替代动作
    """


class microsoft_graphrag_query(BaseModel):
    """健身 GraphRAG 知识推理工具

    当用户提出需要多跳推理、综合分析的健身问题时使用。
    """

    query: str = Field(..., description="需要通过 GraphRAG 进行深度推理的复杂健身问题")


class text2sql_query(BaseModel):
    """健身数据结构化查询工具

    用于健身日志、训练记录、用户数据等关系型数据库查询。
    """

    task: str = Field(..., description="需要执行的健身数据查询任务描述")

    connection_id: Optional[int] = Field(
        default=None, description="数据库连接配置 ID，留空则使用默认连接"
    )

    db_type: str = Field(
        default="MySQL", description="数据库类型，如 MySQL、PostgreSQL"
    )

    max_rows: int = Field(default=1000, description="结果最大返回行数")

    connection_string: Optional[str] = Field(
        default=None, description="直接传入数据库连接字符串（优先级最高）"
    )
