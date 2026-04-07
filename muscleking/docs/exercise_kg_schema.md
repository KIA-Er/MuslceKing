# Exercise Knowledge Graph Schema (健身动作知识图谱模式)

## Node Types (节点类型)

| Label | 中文含义 | 主要属性 | 备注示例 |
|-------|----------|----------|----------|
| Exercise | 训练动作 | name (唯一), description, calories_per_min | 深蹲、卧推等 |
| Muscle | 肌群/肌肉 | name (唯一) | 胸肌、股四头肌等 |
| Equipment | 器械 | name (唯一) | 哑铃、杠铃、自重等 |
| Difficulty | 难度等级 | name (唯一) | easy, medium, hard |
| TrainingGoal | 训练目标 | name (唯一) | 增肌、减脂、力量等 |
| ExerciseStep | 动作步骤 | order, instruction | 步骤序号+说明 |
| InjuryRisk | 伤病风险 | name (唯一) | 膝盖压力、腰椎受力等 |
| Benefit | 训练收益 | name (唯一) | 肌肥大、心肺提升等 |

## Relationship Types (关系类型)

| 类型 | 起点 → 终点 | 中文含义 | 属性 |
|------|------------|----------|------|
| TARGETS_MUSCLE | Exercise → Muscle | 动作目标肌群 | - |
| USES_EQUIPMENT | Exercise → Equipment | 动作使用器械 | - |
| HAS_DIFFICULTY | Exercise → Difficulty | 动作难度等级 | - |
| SUPPORTS_GOAL | Exercise → TrainingGoal | 动作支持目标 | - |
| HAS_STEP | Exercise → ExerciseStep | 动作包含步骤 | order (序号) |
| HAS_RISK | Exercise → InjuryRisk | 动作存在风险 | - |
| HAS_BENEFIT | Exercise → Benefit | 动作产生收益 | - |
| INCLUDES_EXERCISE | WorkoutPlan → Exercise | 计划包含动作 | sets, reps, weight, rest_time |

## Property Notes (属性说明)

### Exercise 节点属性
- **name**: 动作名称（必须唯一，建议使用中文常用名称）
- **description**: 动作的标准描述和注意事项
- **calories_per_min**: 每分钟消耗的卡路里（数值）

### Muscle 节点属性
- **name**: 肌群名称（标准解剖学名称或常用健身术语）
- **category**: 肌群分类（推肌群、拉肌群、腿部等）

### ExerciseStep 关系属性
- **order**: 步骤执行顺序（整数，从1开始）
- **instruction**: 步骤详细说明

## 示例 Cypher Query

```cypher
// 查询深蹲的完整信息
MATCH (e:Exercise {name: "深蹲"})
OPTIONAL MATCH (e)-[:TARGETS_MUSCLE]->(m:Muscle)
OPTIONAL MATCH (e)-[:USES_EQUIPMENT]->(eq:Equipment)
OPTIONAL MATCH (e)-[:HAS_DIFFICULTY]->(d:Difficulty)
OPTIONAL MATCH (e)-[:SUPPORTS_GOAL]->(g:TrainingGoal)
RETURN e.name AS 动作名称,
       collect(DISTINCT m.name) AS 目标肌群,
       collect(DISTINCT eq.name) AS 器械,
       collect(DISTINCT d.name) AS 难度,
       collect(DISTINCT g.name) AS 训练目标
```

## 数据来源

本知识图谱数据来源于 exerciseDB API，包含以下字段映射：

| exerciseDB 字段 | 知识图谱节点/关系 |
|-----------------|-------------------|
| name | Exercise.name |
| equipment | Equipment (USES_EQUIPMENT) |
| bodyPart | Muscle (TARGETS_MUSCLE) |
| targetMuscle | Muscle (TARGETS_MUSCLE) |
| secondaryMuscles | Muscle (TARGETS_MUSCLE) |
| instructions | ExerciseStep (HAS_STEP) |
| gifUrl | 暂不导入 |
| id | 保留作为外部引用 |

**End Patch**
