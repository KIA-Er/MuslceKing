import re
from muscleking.app.agents.retrieve.base import BaseCypherExampleRetriever
from muscleking.app.services.llm_client import LLMClient


class FitnessCypherRetriever(BaseCypherExampleRetriever):
    """
    健身场景的 Cypher 示例检索器
    使用大模型生成 Cypher 查询，为 LLM 提供高质量的健身查询样本
    """

    def __init__(self, llm_client: LLMClient = None):
        super().__init__()
        object.__setattr__(self, "_llm_client", llm_client or LLMClient())

        # 预定义健身 Cypher 示例（用于 few-shot）
        self._predefined_examples = [
            {
                "question": "深蹲怎么做？",
                "cypher": """MATCH (e:Exercise {name: '深蹲'})
RETURN e.description AS 动作说明""",
            },
            {
                "question": "深蹲是什么难度？",
                "cypher": """MATCH (e:Exercise {name: '深蹲'})-[:HAS_DIFFICULTY]->(d:Difficulty)
RETURN d.name AS 难度""",
            },
            {
                "question": "引体向上需要什么器械？",
                "cypher": """MATCH (e:Exercise {name: '引体向上'})-[:USES_EQUIPMENT]->(eq:Equipment)
RETURN collect(eq.name) AS 器械""",
            },
            {
                "question": "练胸肌的动作有哪些？",
                "cypher": """MATCH (e:Exercise)-[:TARGETS_MUSCLE]->(m:Muscle {name: '胸肌'})
RETURN e.name AS 动作名称 LIMIT 15""",
            },
            {
                "question": "新手适合哪些动作？",
                "cypher": """MATCH (e:Exercise)-[:HAS_DIFFICULTY]->(d:Difficulty {name: '初级'})
RETURN e.name AS 动作名称 LIMIT 15""",
            },
            {
                "question": "适合减脂的动作有哪些？",
                "cypher": """MATCH (e:Exercise)-[:SUPPORTS_GOAL]->(g:TrainingGoal {name: '减脂'})
RETURN e.name AS 动作名称 LIMIT 15""",
            },
            {
                "question": "深蹲练哪些肌肉？",
                "cypher": """MATCH (e:Exercise {name: '深蹲'})-[:TARGETS_MUSCLE]->(m:Muscle)
RETURN m.name AS 肌群""",
            },
            {
                "question": "深蹲的完整动作步骤",
                "cypher": """MATCH (e:Exercise {name: '深蹲'})-[:HAS_STEP]->(s:ExerciseStep)
RETURN s.order AS 步骤序号, s.instruction AS 步骤说明
ORDER BY s.order""",
            },
        ]

    # ==================== Prompt ====================

    def _get_llm_generation_prompt(self, query: str) -> str:
        return f"""你是一个专业的 Neo4j Cypher 查询生成专家，专注于【健身知识图谱】。
请根据用户的自然语言问题生成准确、安全、只读的 Cypher 查询。

【健身知识图谱 Schema】

节点类型：
- Exercise: 动作 (name, description)
- Muscle: 肌群 (name)
- Equipment: 器械 (name)
- Difficulty: 难度 (name)
- TrainingGoal: 训练目标 (name)
- WorkoutPlan: 训练计划 (name)
- ExerciseStep: 动作步骤 (order, instruction)
- InjuryRisk: 伤病风险 (name)
- Benefit: 训练收益 (name)

关系类型：
- TARGETS_MUSCLE
- USES_EQUIPMENT
- HAS_DIFFICULTY
- SUPPORTS_GOAL
- INCLUDES_EXERCISE
- HAS_STEP
- HAS_RISK
- HAS_BENEFIT

【示例】
{self._format_examples_for_prompt()}

现在请为以下问题生成 Cypher 查询：
问题：{query}

要求：
1. 只返回 Cypher 查询语句
2. 不要包含解释、注释或 Markdown
3. 如果问题不明确，返回一个通用 MATCH 查询
"""

    def _format_examples_for_prompt(self) -> str:
        texts = []
        for i, ex in enumerate(self._predefined_examples[:5], 1):
            texts.append(f"{i}. 问题：{ex['question']}\n   Cypher：{ex['cypher']}")
        return "\n\n".join(texts)

    # ==================== LLM 生成 ====================

    async def _generate_cypher_with_llm(self, query: str) -> str:
        try:
            prompt = self._get_llm_generation_prompt(query)
            response = await self._llm_client.chat(
                system_prompt="你是一个严格遵循示例的 Neo4j Cypher 生成专家。",
                user_message=prompt,
                temperature=0.1,
            )

            cypher = response.strip()

            if cypher.startswith("```"):
                cypher = cypher.split("\n", 1)[1]
            if cypher.endswith("```"):
                cypher = cypher.rsplit("\n", 1)[0]

            return cypher.strip()
        except Exception as e:
            print(f"LLM 生成 Cypher 失败: {e}")
            return ""

    # ==================== Async 示例接口 ====================

    async def get_examples_async(self, query: str, k: int = 5) -> str:
        selected_examples = []

        llm_cypher = await self._generate_cypher_with_llm(query)
        if llm_cypher:
            selected_examples.append({"question": query, "cypher": llm_cypher})

        selected_examples.extend(self._predefined_examples)

        def compute_relevance(example, query_text):
            score = 0
            q_words = set(re.findall(r"\w+", query_text.lower()))
            e_words = set(re.findall(r"\w+", example["question"].lower()))

            score += len(q_words & e_words) * 2

            keywords = {
                "怎么做": 3,
                "步骤": 3,
                "肌肉": 2,
                "练": 2,
                "难度": 2,
                "器械": 2,
                "风险": 2,
                "收益": 2,
                "计划": 2,
                "减脂": 2,
                "增肌": 2,
            }

            for k, w in keywords.items():
                if k in query_text and k in example["question"]:
                    score += w

            if example["question"] == query:
                score += 100

            return score

        scored = [(ex, compute_relevance(ex, query)) for ex in selected_examples]
        scored.sort(key=lambda x: x[1], reverse=True)
        final_examples = [ex for ex, _ in scored[:k]]

        return "\n\n".join(
            f"Question: {ex['question']}\nCypher: {ex['cypher']}"
            for ex in final_examples
        )

    # ==================== 同步版本（向后兼容） ====================

    def get_examples(self, query: str, k: int = 5) -> str:
        selected_examples = self._predefined_examples.copy()

        def compute_relevance(example, query_text):
            score = 0
            q_words = set(re.findall(r"\w+", query_text.lower()))
            e_words = set(re.findall(r"\w+", example["question"].lower()))
            score += len(q_words & e_words) * 2
            return score

        scored = [(ex, compute_relevance(ex, query)) for ex in selected_examples]
        scored.sort(key=lambda x: x[1], reverse=True)
        final_examples = [ex for ex, _ in scored[:k]]

        return "\n\n".join(
            f"Question: {ex['question']}\nCypher: {ex['cypher']}"
            for ex in final_examples
        )
