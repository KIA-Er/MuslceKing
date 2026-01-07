# WGER Knowledge Graph Schema (wger 健身知识图谱模式)

基于 https://github.com/wger-project/wger 的数据结构设计

## 数据源说明

WGER 提供以下 API 端点获取健身数据：
- `exercises/info` - 获取练习详情
- `exercises/search` - 搜索练习
- `muscles` - 获取肌群列表
- `equipment` - 获取器械列表
- `categories` - 获取练习类别
- `licenses` - 获取许可信息

## Node Types (节点类型)

| Label | 中文含义 | 主要属性 | 备注示例 |
|-------|----------|----------|----------|
| Exercise | 训练动作 | id, name (唯一), description, created | 深蹲、卧推等 |
| Muscle | 肌群 | id, name (唯一), is_front (是否正面) | 胸肌、股四头肌等 |
| Equipment | 器械 | id, name (唯一) | 哑铃、杠铃、自重等 |
| Category | 练习类别 | id, name (唯一) | 力量训练、有氧、拉伸等 |
| License | 许可信息 | id, name, url | CC-BY-SA 等 |
| RepetitionUnit | 重复单位 | id, name | 次、秒 |
| WeightUnit | 重量单位 | id, name | kg、lbs |

## Relationship Types (关系类型)

| 类型 | 起点 → 终点 | 中文含义 | 属性 |
|------|------------|----------|------|
| TARGETS_MUSCLE | Exercise → Muscle | 动作目标肌群 | is_main (是否主肌群) |
| USES_EQUIPMENT | Exercise → Equipment | 动作使用器械 | - |
| BELONGS_TO_CATEGORY | Exercise → Category | 动作所属类别 | - |
| HAS_LICENSE | Exercise → License | 动作许可信息 | - |
| HAS_REPETITION_UNIT | Exercise → RepetitionUnit | 重复单位 | - |
| HAS_WEIGHT_UNIT | Exercise → WeightUnit | 重量单位 | - |

## 字段映射 (wger API → 知识图谱)

### Exercise 节点
| wger 字段 | 知识图谱属性 | 说明 |
|-----------|-------------|------|
| id | id | 动作唯一标识 |
| name | name | 动作名称 |
| description | description | 动作描述 (可能包含HTML) |
| created | created | 创建时间 |
| updated | updated | 更新时间 |
| uuid | uuid | 外部唯一标识 |

### Muscle 节点
| wger 字段 | 知识图谱属性 | 说明 |
|-----------|-------------|------|
| id | id | 肌群唯一标识 |
| name | name | 肌群名称 (英文) |
| name_en | name_en | 英文名称 |
| is_front | is_front | 是否正面肌群 |

### Equipment 节点
| wger 字段 | 知识图谱属性 | 说明 |
|-----------|-------------|------|
| id | id | 器械唯一标识 |
| name | name | 器械名称 (英文) |

### Category 节点
| wger 字段 | 知识图谱属性 | 说明 |
|-----------|-------------|------|
| id | id | 类别唯一标识 |
| name | name | 类别名称 |

## 示例 Cypher Query

### 查询动作及其关联信息
```cypher
MATCH (e:Exercise {id: $exercise_id})
OPTIONAL MATCH (e)-[:TARGETS_MUSCLE]->(m:Muscle)
OPTIONAL MATCH (e)-[:USES_EQUIPMENT]->(eq:Equipment)
OPTIONAL MATCH (e)-[:BELONGS_TO_CATEGORY]->(c:Category)
RETURN e.name AS 动作名称,
       collect(DISTINCT m.name) AS 目标肌群,
       collect(DISTINCT eq.name) AS 使用器械,
       c.name AS 类别
```

### 查询某肌群的所有动作
```cypher
MATCH (m:Muscle {name: $muscle_name})<-[:TARGETS_MUSCLE]-(e:Exercise)
RETURN e.name AS 动作名称, e.description AS 描述
ORDER BY e.name
```

### 查询使用某器械的所有动作
```cypher
MATCH (eq:Equipment {name: $equipment_name})<-[:USES_EQUIPMENT]-(e:Exercise)
RETURN e.name AS 动作名称, e.description AS 描述
```

## WGER API 响应示例

### Exercise 响应
```json
{
    "id": 1,
    "name": "Barbell Squat",
    "description": "<p>The barbell squat is one of the...</p>",
    "created": "2013-07-05T16:12:32.247263Z",
    "updated": "2020-05-15T15:51:47.783545Z",
    "uuid": "c8e2d5f5-6d3c-4a9f-8f4a-4e2d5f5c8e2",
    "muscles": [
        {"id": 11, "name": "Quadriceps", "is_front": true},
        {"id": 12, "name": "Glutes", "is_front": false}
    ],
    "muscles_secondary": [
        {"id": 13, "name": "Hamstrings", "is_front": false}
    ],
    "equipment": [
        {"id": 1, "name": "Barbell"},
        {"id": 8, "name": "Squat rack"}
    ],
    "category": {"id": 10, "name": "Strength"},
    "license": {"id": 1, "name": "CC BY-SA", "url": "https://creativecommons.org/licenses/by-sa/4.0/"},
    "license_author": "wger.de",
    "images": [],
    "videos": [],
    "comments": []
}
```

### Muscle 响应
```json
{
    "id": 1,
    "name": "Pectoralis major",
    "name_en": "Pectoralis major",
    "is_front": true
}
```

### Equipment 响应
```json
{
    "id": 1,
    "name": "Barbell"
}
```

## Property Notes (属性说明)

### Exercise 节点属性
- **id**: 动作在 wger 数据库中的唯一标识
- **name**: 动作名称（英文，建议保留原文）
- **description**: 动作描述，可能包含 HTML 标签
- **created**: 创建时间 (ISO 8601 格式)
- **updated**: 更新时间 (ISO 8601 格式)
- **uuid**: wger 的全局唯一标识符

### Muscle 节点属性
- **id**: 肌群唯一标识
- **name**: 肌群名称（英文）
- **is_front**: 是否为正面肌群 (true/false)

**End Patch**