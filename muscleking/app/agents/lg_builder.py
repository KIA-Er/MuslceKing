"""
langgraph 多路由图构造
"""
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger
from muscleking.app.models.model_agents import AdditionalGuardrailsOutput
from dataclasses import dataclass, field
from muscleking.app.agents.lg_states import AgentState, InputState, Router, GradeHallucinations
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from muscleking.config import settings










logger = logger.bind(service="lg_builder")


# 定义状态图
builder = StateGraph(AgentState, input=InputState)
# 添加节点
builder.add_node(analyze_and_route_query) # 意图识别
builder.add_node(respond_to_general_query)#默认回复
# builder.add_node(get_additional_info) # 图结构信息
# builder.add_node("create_research_plan", create_research_plan)  # 这里是graphrag neo4j-query
# builder.add_node(create_image_query)
# builder.add_node(create_file_query)
# builder.add_node(create_kb_query)

# 添加边
builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)




#意图识别
async def analyze_and_route_query(
        state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """

    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured for router analysis.")

    model = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model_name=settings.OPENAI_MODEL,
        openai_api_base=settings.OPENAI_API_BASE,
        temperature=0.7,
        tags=["router"],
    )

    # 拼接提示模版 + 用户的实时问题（包含历史上下文对话）
    messages = [
                   {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
               ] + state.messages
    logger.info("-----Analyze user query type-----")
    logger.info(f"History messages: {state.messages}")

    question_text = state.messages[-1].content if state.messages else ""
    heuristic_router = _heuristic_router(question_text)
    fallback_router: Router = heuristic_router or Router(
        type="kb-query",
        logic="fallback: default to knowledge base routing",
        question=question_text,
    )

    allowed_types: set[str] = {
        "general-query",
        "additional-query",
        "kb-query",
        "graphrag-query",
        "image-query",
        "file-query",
        "text2sql-query",
    }

    try:
        raw_response = await model.with_structured_output(Router).ainvoke(messages)
    except Exception as exc:
        logger.warning("Router LLM failed: %s. Falling back to KB query.", exc)
        return {"router": fallback_router}

    response = raw_response if isinstance(raw_response, Router) else Router.model_validate(raw_response)
    router_type = response.type
    logic = response.logic or ""

    if not router_type or router_type not in allowed_types:
        logger.warning(
            "Router returned invalid type `%s`; applying heuristic fallback.", router_type
        )
        heuristic_router = _heuristic_router(question_text)
        if heuristic_router:
            sanitized = heuristic_router
            if not sanitized.logic:
                sanitized.logic = logic or ""
            return {"router": sanitized}
        return {
            "router": Router(
                type="kb-query",
                logic=logic or "fallback: invalid router output",
                question=question_text,
            )
        }

    sanitized_router = Router(
        type=router_type,
        logic=logic,
        question=response.question or question_text,
        decision=response.decision,
        confidence=response.confidence,
        reasoning=response.reasoning,
    )

    # Heuristic router is only used when the LLM output is invalid (handled above).
    logger.info(f"Analyze user query type completed, result: {sanitized_router}")
    return {"router": sanitized_router}

