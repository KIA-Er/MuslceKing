"""
langgraph 多路由图构造
"""
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger
from muscleking.app.models.model_lg_state import AdditionalGuardrailsOutput
from dataclasses import dataclass, field
from muscleking.app.models.model_lg_state import AgentState, InputState, Router, GradeHallucinations
from muscleking.app.agents.lg_prompts import (ROUTER_SYSTEM_PROMPT,GENERAL_QUERY_SYSTEM_PROMPT,GUARDRAILS_SYSTEM_PROMPT,GET_ADDITIONAL_SYSTEM_PROMPT)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from muscleking.config import settings
from typing import cast, Literal, List, Dict, Any, Optional
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from muscleking.app.persistence.core.neo4jconn import get_neo4j_graph
from muscleking.app.utils.utils import retrieve_and_parse_schema_from_graph_for_prompts

logger = logger.bind(service="lg_builder")


# 意图识别：llm路由
async def analyze_and_route_query(
        state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """用llm分析用户query并选择适当的路由。

    Args:
        state (AgentState): Agent当前的状态信息,包含历史对话等信息.
        config (RunnableConfig): 帮助模型进行query分析的配置.

    Returns:
        dict[str, Router]: 包含路由信息的字典,key为"router", value为Router对象.
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
    prompt = [
                   {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
               ] + state.messages
    logger.info("-----分析用户的query类型-----")
    logger.info(f"历史消息记录（包含用户query）: {state.messages}")
    # 提取用户的query
    question_text = state.messages[-1].content if state.messages else ""
    # 启发式路由（仅关键词识别）
    heuristic_router = _heuristic_router(question_text)
    # 当llm路由失败时，设定启发式路由作为后备选项，默认路由（知识库查询）兜底
    fallback_router: Router = heuristic_router or Router(
        type="kb-query",
        logic="fallback: default to knowledge base routing",
        question=question_text,
    )

    allowed_types: set[str] = {
        "general-query",
        "additional-query",
        "kb-query",
        "lightrag-query",
        # "image-query",
        # "file-query",
        # "text2sql-query",
    }

    try:
        raw_response = await model.with_structured_output(Router).ainvoke(prompt)
    except Exception as exc:
        logger.warning("llm路由失败: %s. 使用fallback_router.", exc)
        return {"router": fallback_router}
    # 规范化输出,保证字段访问兼容response是Router类型
    response = raw_response if isinstance(raw_response, Router) else Router.model_validate(raw_response)
    router_type = response.type
    logic = response.logic or ""
    # 路由类型无效时的处理
    if not router_type or router_type not in allowed_types:
        logger.warning(
            "llm路由返回无效的类型: `%s`; 使用启发式路由回退方案.", router_type
        )
        fallback_router.logic = logic or ""
        return {"router": fallback_router}
    # llm路由成功，返回规范化的Router对象
    sanitized_router = Router(
        type=router_type,
        logic=logic,
        question=response.question or question_text,
        decision=response.decision,
        confidence=response.confidence,
        reasoning=response.reasoning,
    )

    logger.info(f"分析用户query并选择适当的路由完成: {sanitized_router}")
    return {"router": sanitized_router}




# 意图识别：启发式路由
def _heuristic_router(question: str) -> Optional[Router]:
    """基于关键词的启发式路由,用于分类用户query.(只写了lightrag-query和general-query的关键词)"""
    if not question:
        return None

    lowered = question.lower()

    # === lightrag 关键词：动作要点 / 训练步骤 / 计划分解 ===
    lightrag_keywords = [
        "怎么练","如何练","怎么做",
        "如何做","动作","要点","姿势","计划","训练计划",
    ]

    general_keywords = [
        "天气", "笑话", "故事", "翻译", "怎么写代码", "调试",
         "推荐电影", "推荐书", "如何学习", "考试",
    ]

    # --- 匹配 lightrag ---
    if any(keyword in lowered for keyword in lightrag_keywords):
        return Router(
            type="lightrag-query",
            logic="keyword fallback: lightrag",
            question=question,
        )
    # --- 匹配 general ---
    if any(keyword in lowered for keyword in general_keywords):
        return Router(
            type="general-query",
            logic="keyword fallback: general",
            question=question,
        )

    return None


# 根据意图识别路由到不同处理节点
def route_query(
        state: AgentState,
) -> Literal[
    "respond_to_general_query", "get_additional_info", "create_research_plan", "create_image_query", "create_file_query", "create_kb_query"]:
    """根据查询分类确定下一步操作。

    Args:
        state (AgentState): 当前代理状态，包括路由器的分类。

    Returns:
        Literal["respond_to_general_query", "get_additional_info", "create_research_plan", "create_image_query", "create_file_query"，"create_kb_query"]: 下一步操作。
    """
    router = _ensure_router(getattr(state, "router", None), fallback_question=state.messages[-1].content if state.messages else "")
    state.router = router
    _type = router.type or "kb-query"

    # 检查配置中是否有图片或文件路径，如果有，优先对应处理
    # if hasattr(state, "config") and state.config:
    #     cfg = state.config.get("configurable", {})
    #     if cfg.get("image_path"):
    #         logger.info("检测到图片路径，转为图片查询处理")
    #         return "create_image_query"
    #     if cfg.get("file_path"):
    #         logger.info("检测到文件路径，转为文件上传处理")
    #         return "create_file_query"

    if _type == "general-query":
        return "respond_to_general_query"
    elif _type == "additional-query":
        return "get_additional_info"
    elif _type in ("lightrag-query", "text2sql-query"):  # 图查询或结构化问数
        return "create_research_plan"
    # elif _type == "image-query":
    #     return "create_image_query"
    # elif _type == "file-query":
    #     return "create_file_query"
    elif _type=="kb-query":
        return "create_kb_query"
    else:
        raise ValueError(f"Unknown router type {_type}")


# 将任意 router 结构转化成 Router对象
def _ensure_router(router_obj: Any, *, fallback_question: str = "") -> Router:
    """将任意 router 结构转换为 Router 模型，保持字段访问兼容。"""
    if isinstance(router_obj, Router):
        return router_obj
    if isinstance(router_obj, dict):
        try:
            return Router.model_validate(router_obj)
        except Exception:
            pass
    return Router(type="kb-query", logic="missing router", question=fallback_question)

# 从 Config 中提取 configurable 字段
def _extract_configurable(config: Any) -> Dict[str, Any]:
    """提取 LangGraph RunnableConfig 中的 configurable 字段，确保返回字典。"""
    if not config:
        return {}
    if isinstance(config, dict):
        value = config.get("configurable", {})
        return value if isinstance(value, dict) else {}
    logger.warning("无法从 config 中提取 configurable 字段,返回空字典。请确保config和里面的configurable字段都必须是字典。")
    return {}
    

#类型一：直接用大模型回答不含本地及其他外部知识
async def respond_to_general_query(
        state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """生成对一般查询的响应，完全基于大模型，不会触发任何外部服务的调用，包括自定义工具、知识库查询等。
    当路由器将查询分类为一般问题时，将调用此节点。
    Args:
        state (AgentState): 当前代理状态，包括对话历史和路由逻辑。
        config (RunnableConfig): 用于配置响应生成的模型。
    Returns:
        Dict[str, List[BaseMessage]]: 包含'messages'键的字典，其中包含生成的响应。
    """
    logger.info("-----generate general-query response-----")

    # 使用大模型生成回复
    model = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY, model_name=settings.OPENAI_MODEL,
                       openai_api_base=settings.OPENAI_API_BASE, temperature=0.7,
                       tags=["general_query"])

    router = _ensure_router(getattr(state, "router", None), fallback_question=state.messages[-1].content if state.messages else "")
    state.router = router
    system_prompt = GENERAL_QUERY_SYSTEM_PROMPT.format(
        logic=router.logic
    )

    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}



# 类型二：需要从用户获取更多信息再回答
async def get_additional_info(
        state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """生成一个响应，要求用户提供更多信息。

    当路由确定需要从用户那里获取更多信息时，将调用此函数。

    Args:
        state (AgentState): 当前代理状态，包括对话历史和路由逻辑。
        config (RunnableConfig): 用于配置响应生成的模型。

    Returns:
        Dict[str, List[BaseMessage]]: 包含'messages'键的字典，其中包含生成的响应。
    """
    logger.info("------continue to get additional info------")

    # 使用大模型生成回复
    model = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY, model_name=settings.OPENAI_MODEL,
                       openai_api_base=settings.OPENAI_API_BASE, temperature=0.7,
                       tags=["additional_info"])
    # 如果用户的问题是健身相关，但与自己的业务无关，则需要返回"无关问题"

    # 首先连接 Neo4j 图数据库
    try:
        neo4j_graph = get_neo4j_graph()
        logger.info("success to get Neo4j graph database connection")
    except Exception as e:
        logger.error(f"failed to get Neo4j graph database connection: {e}")
        neo4j_graph = None

    # 定义健身助手服务范围（用户友好的业务描述）
    scope_description = """
    健身智能助手服务范围：为您提供全方位的运动指导和健康知识，包括但不限于：

    💪 健身训练与动作指导
    - 各类力量训练、心肺训练、柔韧性训练的详细方法
    - 动作标准、训练次数、组数、休息时间
    - 分步骤动作演示和训练小贴士

    🥗 营养饮食与补剂建议
    - 健身相关营养知识（蛋白质、碳水、脂肪等摄入建议）
    - 健康饮食搭配与餐前餐后建议
    - 常见健身补剂的使用方法与注意事项

    🏋️‍♂️ 器械与训练计划
    - 家庭健身器械或健身房器械使用技巧
    - 个性化训练计划制定建议
    - 不同目标（增肌、减脂、塑形）的训练方案

    🧘 身体健康与运动恢复
    - 拉伸、放松和康复训练方法
    - 运动前热身与运动后恢复技巧
    - 特殊人群（孕期、老年人、慢性病人）的运动注意事项

    暂不支持：政治、娱乐八卦、新闻时事、天气预报、网购推荐、医疗诊断等非健身健康相关内容。
    如遇此类问题，我会礼貌地引导您回到运动健身话题～
    """

    scope_context = (
        f"参考此范围描述来决策:\n{scope_description}"
        if scope_description is not None
        else ""
    )

    # 动态从 Neo4j 图表中获取图表结构
    graph_context = (
        f"\n参考图表结构来回答:\n{retrieve_and_parse_schema_from_graph_for_prompts(neo4j_graph)}"
        if neo4j_graph is not None
        else ""
    )

    message = scope_context + graph_context + "\nQuestion: {question}"

    # 拼接提示模版
    full_system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                GUARDRAILS_SYSTEM_PROMPT,
            ),
            (
                "human",
                (message),
            ),
        ]
    )

    # 构建格式化输出的 Chain， 如果匹配，返回 continue，否则返回 end
    guardrails_chain = full_system_prompt | model.with_structured_output(AdditionalGuardrailsOutput)
    guardrails_output: AdditionalGuardrailsOutput = await guardrails_chain.ainvoke(
        {"question": state.messages[-1].content if state.messages else ""}
    )

    # 空值检查：如果 LLM 返回 None，默认为 proceed
    if guardrails_output is None:
        logger.warning("Guardrails returned None, defaulting to proceed")
        guardrails_output = AdditionalGuardrailsOutput(decision="proceed")

    # 根据格式化输出的结果，返回不同的响应
    if guardrails_output.decision == "end":
        logger.info("-----Fail to pass guardrails check-----")
        return {"messages": [AIMessage(content="厨友您好～抱歉哦，这个问题不太属于我们的菜谱范围呢，我主要帮您解答菜谱和烹饪方面的问题～😊")]}
    else:
        logger.info("-----Pass guardrails check-----")
        router = _ensure_router(getattr(state, "router", None), fallback_question=state.messages[-1].content if state.messages else "")
        state.router = router
        system_prompt = GET_ADDITIONAL_SYSTEM_PROMPT.format(
            logic=router.logic
        )
        messages = [{"role": "system", "content": system_prompt}] + state.messages
        response = await model.ainvoke(messages)
        return {"messages": [response]}


# 类型三：知识库问答
async def create_kb_query(
        state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """通过一个多工具子图进行知识库查询."""
    logger.info("------execute KB multi-tool query------")
    # 提取用户的query
    last_message = state.messages[-1].content if state.messages else ""
    if not last_message.strip():
        return {"messages": [AIMessage(content="请告诉我具体的问题，我才能帮您查询知识库。")]}
    # 提取config中的configurable字段
    config_opts = _extract_configurable(config)
    kb_top_k = config_opts.get("kb_top_k") or settings.KB_TOP_K
    kb_similarity_threshold = (
        config_opts.get("kb_similarity_threshold")
        if config_opts.get("kb_similarity_threshold") is not None
        else settings.KB_SIMILARITY_THRESHOLD
    )
    kb_filter_expr = config_opts.get("kb_filter_expr")

    knowledge_service: Optional[KnowledgeBaseService] = None
    try:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured for KB multi-tool workflow.")

        llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.OPENAI_MODEL,
            openai_api_base=settings.OPENAI_API_BASE,
            temperature=0.3,
            tags=["kb_multi_tool"],
        )

        knowledge_service = KnowledgeBaseService()

        external_url = settings.KB_EXTERNAL_SEARCH_URL
        if not external_url and settings.INGEST_SERVICE_URL:
            external_url = f"{settings.INGEST_SERVICE_URL.rstrip('/')}/api/search"

        workflow = create_kb_multi_tool_workflow(
            llm=llm,
            knowledge_service=knowledge_service,
            top_k=kb_top_k,
            similarity_threshold=kb_similarity_threshold,
            filter_expr=kb_filter_expr,
            allow_external=settings.KB_ENABLE_EXTERNAL_SEARCH,
            external_search_url=external_url,
        )

        history_payload = [
            {
                "role": getattr(msg, "type", "user"),
                "content": getattr(msg, "content", ""),
            }
            for msg in state.messages[:-1]
            if getattr(msg, "content", "").strip()
        ]

        response = await workflow.ainvoke(
            {
                "question": last_message,
                "history": history_payload,
            }
        )
        answer_text = response.get("answer") or "检索完成，但暂时没有可以分享的结果。"
        sources = response.get("sources", [])

        # 创建包含sources的AIMessage
        ai_message = AIMessage(content=answer_text)
        # 将sources附加到消息的additional_kwargs中
        ai_message.additional_kwargs["sources"] = sources

        return {"messages": [ai_message], "sources": sources}
    except Exception as exc:
        logger.warning("KB multi-tool workflow unavailable (%s); falling back to direct search.", exc)

    # Fallback: direct KB query
    if knowledge_service is None:
        knowledge_service = KnowledgeBaseService()
    knowledge_node = create_knowledge_query_node(knowledge_service=knowledge_service)
    input_state: KnowledgeQueryInputState = {
        "task": last_message,
        "context": {
            "top_k": kb_top_k,
            "similarity_threshold": kb_similarity_threshold,
            "filter_expr": kb_filter_expr,
        },
        "steps": ["kb_query"],
    }
    result = await knowledge_node(input_state)
    answer_text = result.get("answer", "") or "抱歉，我暂时无法从知识库中找到答案。"
    return {"messages": [AIMessage(content=answer_text)]}



checkpointer = MemorySaver()

# 定义状态图
builder = StateGraph(AgentState, input=InputState)
# 添加节点
builder.add_node(analyze_and_route_query) # 意图识别

builder.add_node(respond_to_general_query)# 默认回复
builder.add_node(get_additional_info) # 额外信息
builder.add_node(create_research_plan)  # 这里是lightrag neo4j-query
builder.add_node(create_kb_query)
# builder.add_node(create_image_query)
# builder.add_node(create_file_query)

# 添加边
builder.add_edge(START, analyze_and_route_query)
builder.add_conditional_edges(analyze_and_route_query, route_query)

graph = builder.compile(checkpointer=checkpointer)





