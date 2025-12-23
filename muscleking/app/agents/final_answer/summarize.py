"""
Fitness Agent summarization node based on LangGraph.
"""

from typing import Any, Callable, Coroutine, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from muscleking.app.agents.agent_state import OverallState
from muscleking.app.agents.final_answer.summarize import (
    create_summarization_prompt_template,
)

generate_summary_prompt = create_summarization_prompt_template()


def create_summarization_node(
    llm: BaseChatModel,
) -> Callable[[OverallState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Create a Fitness Summarization node for a LangGraph workflow.
    """

    generate_summary = generate_summary_prompt | llm | StrOutputParser()

    async def summarize(state: OverallState) -> Dict[str, Any]:
        """
        Summarize results of fitness-related Cypher queries.
        """
        tasks = state.get("tasks", [])
        cypher_entries = state.get("cyphers", [])

        narrative_sections: List[str] = []
        metric_sections: List[str] = []
        error_sections: List[str] = []

        def _format_rows(rows: List[Dict[str, Any]]) -> str:
            if not rows:
                return ""

            # 单条记录
            if len(rows) == 1:
                row = rows[0]
                return "; ".join(f"{k}：{v}" for k, v in row.items())

            lines: List[str] = []

            # 训练步骤（步骤序号 + 动作说明）
            is_training_steps = all(
                "步骤序号" in row and "动作说明" in row for row in rows if isinstance(row, dict)
            )

            # 动作列表（动作 + 目标肌群）
            is_exercise_list = all(
                "动作" in row and "目标肌群" in row for row in rows if isinstance(row, dict)
            )

            for idx, row in enumerate(rows, 1):
                if is_training_steps:
                    step = row.get("步骤序号", idx)
                    desc = row.get("动作说明", "")
                    lines.append(f"{step}. {desc}")

                elif is_exercise_list:
                    exercise = row.get("动作", "")
                    muscle = row.get("目标肌群", "")
                    equipment = row.get("器械", "")
                    parts = [exercise]
                    if muscle:
                        parts.append(f"目标：{muscle}")
                    if equipment:
                        parts.append(f"器械：{equipment}")
                    lines.append("｜".join(parts))

                else:
                    row_desc = ", ".join(f"{k}：{v}" for k, v in row.items())
                    lines.append(f"{idx}. {row_desc}")

            return "\n".join(lines)

        for idx, cypher in enumerate(cypher_entries):
            if hasattr(cypher, "model_dump"):
                data = cypher.model_dump()
            elif isinstance(cypher, dict):
                data = cypher
            else:
                data = {}

            task_label = tasks[idx].question if idx < len(tasks) else data.get("task", "")

            records = data.get("records") or {}
            errors = data.get("errors") or []

            if errors:
                error_sections.append(
                    f"{task_label}：{'；'.join(errors)}" if task_label else "；".join(errors)
                )
                continue

            if not records:
                continue

            if isinstance(records, dict):
                # 叙述性总结（如：某动作的训练价值）
                if isinstance(records.get("result"), str):
                    narrative_sections.append(records["result"].strip())

                # 数值型答案（如：推荐次数 / 热量）
                answer = records.get("answer")
                if answer:
                    metric_sections.append(f"{task_label}：{answer}".strip())

                rows = records.get("rows")
                if isinstance(rows, list):
                    formatted = _format_rows(rows)
                    if formatted:
                        metric_sections.append(
                            f"{task_label}：\n{formatted}".rstrip()
                            if task_label
                            else formatted
                        )

            elif isinstance(records, list):
                formatted = _format_rows(records)
                if formatted:
                    metric_sections.append(
                        f"{task_label}：\n{formatted}".rstrip()
                        if task_label
                        else formatted
                    )

        sections: List[str] = []

        if narrative_sections:
            sections.append("### 🏋️ 健身内容概览\n" + "\n\n".join(narrative_sections))

        if metric_sections:
            sections.append("### 📊 训练数据\n" + "\n\n".join(metric_sections))

        if error_sections:
            sections.append("### ⚠️ 查询提示\n" + "\n".join(f"- {msg}" for msg in error_sections))

        summary = "\n\n".join(sections).strip() or "暂无可总结的健身数据。"

        return {
            "summary": summary,
            "steps": ["summarize_fitness"],
        }

    return summarize
