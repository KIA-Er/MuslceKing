from typing import Any, Callable, Coroutine, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.base import Runnable
from langchain_neo4j import Neo4jGraph
from loguru import logger
from muscleking.app.agents.guardrails.guardrails_prompt import create_guardrails_prompt_template
from muscleking.app.agents.agent_state import InputState


logger = logger.bind(service="guardrails")


def create_guardrails_node(
    llm: BaseChatModel,
    graph: Optional[Neo4jGraph] = None,
    scope_description: Optional[str] = None,
) -> Callable[[InputState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Create a guardrails node for a Fitness Agent LangGraph workflow.
    """

    guardrails_prompt = create_guardrails_prompt_template(
        graph=graph, scope_description=scope_description
    )

    guardrails_chain: Runnable[Dict[str, Any], Any] = (
        guardrails_prompt | llm.with_structured_output(GuardrailsOutput)
    )

    async def guardrails(state: InputState) -> Dict[str, Any]:
        """
        Decide whether the fitness-related question is in scope.
        """

        question = state.get("question", "")

        heuristics_keywords = [
            "健身",
            "训练",
            "锻炼",
            "力量",
            "有氧",
            "跑步",
            "增肌",
            "减脂",
            "卡路里",
            "热量",
            "消耗",
            "组数",
            "次数",
            "重量",
            "卧推",
            "深蹲",
            "硬拉",
            "心率",
            "体脂",
            "BMI",
            "计划",
            "训练计划",
            "日志",
            "记录",
            "统计",
            "多少",
        ]

        if any(keyword in question for keyword in heuristics_keywords) or "?" in question or "？" in question:#TODO 有bug，“今天天气如何？”能够通过范围筛选
            logger.info(
                "Fitness Guardrails: 命中健身关键词，直接进入 planner。",
                extra={"question": question},
            )
            return {
                "next_action": "planner",
                "summary": None,
                "steps": ["guardrails"],
            }

        try:
            guardrails_output: GuardrailsOutput = await guardrails_chain.ainvoke(
                {"question": question}
            )
        except Exception as exc:
            logger.warning("Fitness Guardrails LLM 调用失败，回退到 planner: %s", exc)
            return {
                "next_action": "planner",
                "summary": None,
                "steps": ["guardrails"],
            }

        decision = guardrails_output.decision
        summary = None

        if decision == "end":
            if any(keyword in question for keyword in heuristics_keywords):
                logger.info(
                    "Fitness Guardrails 兜底触发：问题含健身关键词，强制进入 planner。",
                    extra={"question": question},
                )
                decision = "planner"

        if decision == "end":
            summary = "抱歉，这个问题暂时不在健身助手的服务范围内，可以试试询问训练、计划或数据统计哦～"

        decision_info = {
            "next_action": decision,
            "summary": summary,
            "steps": ["guardrails"],
        }

        logger.info(f"Fitness Guardrails Decision Info: {decision_info}")

        return decision_info

    return guardrails


from typing import Literal
from pydantic import BaseModel, Field


class GuardrailsOutput(BaseModel):
    decision: Literal["end", "planner"] = Field(
        description="Decision on whether the question is related to the graph contents."
    )
