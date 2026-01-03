from typing import Dict, Tuple, List

COLUMN_DESCRIPTIONS: Dict[Tuple[str, str], str] = {

    # ========== 用户表 ==========
    ("users", "id"): "用户唯一标识。",
    ("users", "name"): "用户姓名或昵称。",
    ("users", "gender"): "性别，取值 male/female/other。",
    ("users", "age"): "年龄（岁）。",
    ("users", "height"): "身高（cm）。",
    ("users", "weight"): "体重（kg）。",
    ("users", "goal"): "主要健身目标，如 增肌/减脂/塑形/提升体能。",
    ("users", "experience_level"): "健身经验水平 beginner/intermediate/advanced。",
    ("users", "activity_level"): "日常活动强度 low/medium/high。",
    ("users", "created_at"): "用户创建时间。",

    # ========== 训练计划 ==========
    ("workout_plans", "id"): "训练计划唯一标识。",
    ("workout_plans", "user_id"): "关联用户 users.id 的外键。",
    ("workout_plans", "name"): "训练计划名称。",
    ("workout_plans", "goal"): "该计划目标，如 增肌/减脂。",
    ("workout_plans", "duration_weeks"): "计划持续周数。",
    ("workout_plans", "difficulty"): "训练难度 easy/medium/hard。",
    ("workout_plans", "created_at"): "计划创建时间。",

    # ========== 训练课程 ==========
    ("workouts", "id"): "训练课程唯一标识。",
    ("workouts", "plan_id"): "关联训练计划 workout_plans.id 的外键。",
    ("workouts", "day_of_week"): "训练日（1-7）。",
    ("workouts", "focus"): "训练重点，如 胸/腿/全身/核心。",
    ("workouts", "total_time"): "本次训练总时长（分钟）。",
    ("workouts", "estimated_calories"): "预计消耗热量 (kcal)。",

    # ========== 动作库 ==========
    ("exercises", "id"): "训练动作唯一标识。",
    ("exercises", "name"): "动作名称（唯一）。",
    ("exercises", "muscle_group"): "主要训练肌群，如 胸/背/腿/肩。",
    ("exercises", "equipment"): "所需器械，如 哑铃/杠铃/自重。",
    ("exercises", "difficulty"): "动作难度 easy/medium/hard。",
    ("exercises", "description"): "动作说明与注意事项。",
    ("exercises", "calories_per_min"): "平均每分钟消耗热量 (kcal)。",

    # ========== 课程动作关系 ==========
    ("workout_exercises", "workout_id"): "关联训练课程 workouts.id 的外键。",
    ("workout_exercises", "exercise_id"): "关联动作 exercises.id 的外键。",
    ("workout_exercises", "sets"): "组数。",
    ("workout_exercises", "reps"): "每组次数。",
    ("workout_exercises", "weight"): "负重（kg），可为空。",
    ("workout_exercises", "rest_time"): "组间休息时间（秒）。",
    ("workout_exercises", "order"): "动作执行顺序。",

    # ========== 训练记录 ==========
    ("workout_logs", "id"): "训练记录唯一标识。",
    ("workout_logs", "user_id"): "关联用户 users.id 的外键。",
    ("workout_logs", "workout_id"): "关联训练课程 workouts.id 的外键。",
    ("workout_logs", "date"): "训练日期。",
    ("workout_logs", "actual_time"): "实际训练时长（分钟）。",
    ("workout_logs", "actual_calories"): "实际消耗热量 (kcal)。",
    ("workout_logs", "feeling"): "主观感受，如 轻松/正常/疲劳。",
    ("workout_logs", "notes"): "训练备注。",

    # ========== 体测数据 ==========
    ("body_metrics", "id"): "体测记录唯一标识。",
    ("body_metrics", "user_id"): "关联用户 users.id 的外键。",
    ("body_metrics", "date"): "测量日期。",
    ("body_metrics", "weight"): "体重（kg）。",
    ("body_metrics", "body_fat"): "体脂率 (%)。",
    ("body_metrics", "muscle_mass"): "肌肉量（kg）。",
    ("body_metrics", "bmi"): "BMI 指数。",
    ("body_metrics", "basal_metabolism"): "基础代谢 (kcal/day)。",

    # ========== 饮食记录 ==========
    ("diet_logs", "id"): "饮食记录唯一标识。",
    ("diet_logs", "user_id"): "关联用户 users.id 的外键。",
    ("diet_logs", "date"): "记录日期。",
    ("diet_logs", "total_calories"): "当日摄入总热量 (kcal)。",
    ("diet_logs", "protein"): "蛋白质摄入量 (g)。",
    ("diet_logs", "carbs"): "碳水摄入量 (g)。",
    ("diet_logs", "fat"): "脂肪摄入量 (g)。",
    ("diet_logs", "notes"): "饮食备注。",

}


DOMAIN_SUMMARY = """
- 数据库为健身管理场景，核心实体包含用户 (users)、训练计划 (workout_plans)、训练课程 (workouts)、训练动作 (exercises)、课程动作关系 (workout_exercises)、训练记录 (workout_logs)、体测数据 (body_metrics) 与饮食记录 (diet_logs)。
- workout_plans.user_id -> users.id；workouts.plan_id -> workout_plans.id；
  workout_exercises.workout_id -> workouts.id；workout_exercises.exercise_id -> exercises.id；
  workout_logs.user_id -> users.id；workout_logs.workout_id -> workouts.id；
  body_metrics.user_id -> users.id；diet_logs.user_id -> users.id。
- workout_plans.difficulty 与 exercises.difficulty 的取值限定为 easy/medium/hard；
  users.experience_level 的取值限定为 beginner/intermediate/advanced；
  users.activity_level 的取值限定为 low/medium/high。
- exercises 表仅包含 id、name、muscle_group、equipment、difficulty、description、calories_per_min 等字段；
  所有动作筛选应基于 name、muscle_group 或 equipment，不存在 code 字段。
- workouts.total_time 字段为训练课程总时长（分钟），训练记录的实际时长请使用 workout_logs.actual_time 字段。
- 若需统计某次训练课程的理论消耗热量，可使用 workouts.estimated_calories；
  若需统计用户实际消耗热量，请使用 workout_logs.actual_calories。
- 所有查询均应围绕真实存在的表与字段展开，避免凭空构造表名或字段。
""".strip()


TABLE_DESCRIPTIONS: Dict[str, str] = {

    "users": (
        "用户主表，记录用户的基础信息、身体数据与健身目标。"
        "主键为 id，作为所有健身数据的核心关联对象。"
    ),

    "workout_plans": (
        "训练计划表，描述用户的阶段性健身规划，包含目标、周期、难度等信息。"
        "主键 id，通过 user_id 关联用户。"
    ),

    "workouts": (
        "训练课程表，表示某个训练计划中的具体训练日安排，记录训练重点、总时长与预计消耗。"
        "主键 id，通过 plan_id 关联训练计划。"
    ),

    "exercises": (
        "训练动作库，存储标准化健身动作的名称、目标肌群、器械要求、难度与动作说明。"
    ),

    "workout_exercises": (
        "训练课程与动作的关联表，描述每次训练中包含的具体动作及其执行参数，"
        "包括组数、次数、负重、休息时间与执行顺序。"
        "通过 workout_id 关联训练课程，exercise_id 关联动作库。"
    ),

    "workout_logs": (
        "训练记录表，记录用户每次实际完成训练的情况，包括时长、消耗热量与主观感受等。"
        "通过 user_id 关联用户，通过 workout_id 关联训练课程。"
    ),

    "body_metrics": (
        "体测数据表，用于跟踪用户身体状态变化，包含体重、体脂率、肌肉量、BMI 等关键指标。"
        "通过 user_id 关联用户。"
    ),

    "diet_logs": (
        "饮食记录表，记录用户每日营养摄入情况，包括总热量、蛋白质、碳水和脂肪摄入。"
        "通过 user_id 关联用户。"
    ),

}


RELATIONSHIP_FACTS: List[Dict[str, str]] = [

    {
        "source_table": "workout_plans",
        "source_column": "user_id",
        "target_table": "users",
        "target_column": "id",
        "relationship_type": "many-to-one",
        "description": "每个训练计划隶属于一个用户。",
    },

    {
        "source_table": "workouts",
        "source_column": "plan_id",
        "target_table": "workout_plans",
        "target_column": "id",
        "relationship_type": "many-to-one",
        "description": "训练课程属于某个训练计划，按训练日顺序组织。",
    },

    {
        "source_table": "workout_exercises",
        "source_column": "workout_id",
        "target_table": "workouts",
        "target_column": "id",
        "relationship_type": "many-to-one",
        "description": "每节训练课程由多个训练动作组成，按 order 顺序执行。",
    },

    {
        "source_table": "workout_exercises",
        "source_column": "exercise_id",
        "target_table": "exercises",
        "target_column": "id",
        "relationship_type": "many-to-one",
        "description": "训练动作的基础资料与属性说明。",
    },

    {
        "source_table": "workout_logs",
        "source_column": "user_id",
        "target_table": "users",
        "target_column": "id",
        "relationship_type": "many-to-one",
        "description": "用户的实际训练记录。",
    },

    {
        "source_table": "workout_logs",
        "source_column": "workout_id",
        "target_table": "workouts",
        "target_column": "id",
        "relationship_type": "many-to-one",
        "description": "某次训练记录对应的训练课程。",
    },

    {
        "source_table": "body_metrics",
        "source_column": "user_id",
        "target_table": "users",
        "target_column": "id",
        "relationship_type": "many-to-one",
        "description": "用户体测数据随时间变化记录。",
    },

    {
        "source_table": "diet_logs",
        "source_column": "user_id",
        "target_table": "users",
        "target_column": "id",
        "relationship_type": "many-to-one",
        "description": "用户每日饮食营养摄入记录。",
    },

]