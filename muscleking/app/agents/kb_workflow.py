"""
知识库工作流
"""

from typing import Any, Dict, List, Optional
from muscleking.app.services.knowledge_base_service import KnowledgeBaseService
from langchain_core.language_models import BaseChatModel
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from muscleking.config import settings
from loguru import logger
from muscleking.app.agents.models.kb_state import (
    KBGuardrailsDecision,
    KBRouteDecision,
    KBInputState,
    KBWorkflowState,
    KBOutputState,
)
import aiohttp


kb_logger = logger.bind(service="kb-workflow")


def create_kb_multi_tool_workflow(
    llm: BaseChatModel,
    knowledge_service: Optional[KnowledgeBaseService] = None,
    *,
    top_k: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    filter_expr: Optional[str] = None,
    allow_external: Optional[bool] = None,
    external_search_url: Optional[str] = None,
    external_search_timeout: Optional[float] = None,
    scope_description: Optional[str] = None,
) -> CompiledStateGraph:
    """创建知识库多工具工作流
    包含五个节点:
        1. guardrails:安全护栏节点
        2. kb_router:智能路由节点,支持三种路由策略:local(本地)、external(外部)、hybrid(混合)
        3. local_search:本地检索节点,执行 PostgreSQL 和 Milvus 查询(先执行 PostgreSQL 查询,如果没有结果,再执行 Milvus 查询)
        4. external_search:外部检索节点
        5. finalize:最终处理节点,整合结果并生成答案
    Args:
        llm: 语言模型
        knowledge_service: 知识库服务
        top_k: 检索文档数量
        similarity_threshold: 相似度阈值
        filter_expr: 过滤表达式
        allow_external: 是否允许外部搜索
        external_search_url: 外部搜索URL
        external_search_timeout: 外部搜索超时时间
        scope_description: 知识库作用范围描述,用于判断是否符合知识库作用范围
    Returns:
        CompiledStateGraph: 编译后的状态图
    """
    knowledge_service = knowledge_service or KnowledgeBaseService()
    effective_top_k = top_k or settings.KB_TOP_K
    effective_threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.KB_SIMILARITY_THRESHOLD
    )
    # 外部搜索URL,目前没有配置
    external_url = external_search_url or settings.KB_EXTERNAL_SEARCH_URL
    # 外部搜索是否允许
    allow_external_search = (
        allow_external if allow_external else settings.KB_ENABLE_EXTERNAL_SEARCH
    )
    # 外部搜索URL,如果外部搜索允许但未配置URL,则关闭外部搜索
    if allow_external_search and not external_url:
        kb_logger.warning(
            "External search enabled but KB_EXTERNAL_SEARCH_URL 未配置，已自动关闭外部检索。"
        )
        allow_external_search = False
    # 外部搜索URL,如果外部搜索不允许,则URL设为 None
    if not allow_external_search:
        external_url = None
    # postgreSQL搜索URL
    postgres_search_url = (
        f"{settings.INGEST_SERVICE_URL.rstrip('/')}/api/v1/knowledge/search"
        if settings.INGEST_SERVICE_URL
        else None
    )
    # 外部搜索是否为 PostgreSQL 搜索
    external_is_postgres = bool(
        allow_external_search
        and external_url
        and postgres_search_url
        and external_url.rstrip("/") == postgres_search_url.rstrip("/")
    )
    request_timeout = (
        external_search_timeout
        if external_search_timeout is not None
        else settings.KB_EXTERNAL_SEARCH_TIMEOUT
    )

    scope_text = scope_description or (
        "健身知识库仅用于提供系统性、科普性的健身与运动科学知识，包括但不限于："
        "健身与训练的基础原理（如肌肥大机制、力量增长原理、能量系统）、"
        "运动生理学与训练学相关理论、"
        "营养与饮食科学的基础机制（如蛋白质吸收、能量代谢、宏量与微量营养素作用）、"
        "常见训练方法、训练设备与器械的原理性说明、"
        "运动相关的健康风险、常见损伤机理与一般性运动禁忌的科普解释。"
        "该知识库不提供具体个性化训练计划、饮食方案或处方级建议，"
        "不进行疾病诊断、治疗或医疗建议，"
        "不替代医生、营养师或专业教练的个体化指导。"
        "禁止回答与隐私、政治、成人内容、违法行为或其他未授权主题相关的问题。"
    )

    guardrails_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是企业知识库的安全审查员。服务范围：\n"
                    f"{scope_text}\n\n"
                    "判断用户问题是否位于该范围，并确保不包含违法、隐私或未授权内容。"
                    "若问题不适宜或不在范围内，请返回 decision='end' 并给出中文 summary；"
                    "否则返回 decision='proceed'。"
                ),
            ),
            ("human", "用户问题：{question}"),
        ]
    )
    guardrails_chain = guardrails_prompt | llm.with_structured_output(
        KBGuardrailsDecision
    )

    router_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是健身知识库查询路由器。专门负责将健身、运动科学与营养科普类问题路由到最合适的知识库。\n\n"
                    "## 服务范围\n"
                    '仅限于健身与运动科学的**系统性、科普性知识解释**"。\n\n'
                    "## 本地知识源说明\n"
                    "- **postgres**（PostgreSQL pgvector）：**第一优先级**，存放结构化表格数据、Excel导入的枚举字段\n"
                    "  - 数据更准确、查询更快、覆盖面广\n"
                    "  - 适合：概念解释、机制原理、简要科普说明\n"
                    "  - 典型问题：力量训练如何提高最大力量\n"
                    "  - 执行策略：系统会**先查 postgres**，如果有结果就直接使用，**不会查询 milvus**\n"
                    "- **milvus**（Milvus向量库）：**仅作为兜底**，存放长文本、文章等非结构化内容\n"
                    "  - 只有在 postgres 无结果时才会查询\n"
                    '  - 典型问题："力量训练与有氧训练对能量系统的长期影响"、"肌肥大的完整生理机制详解"\n\n'
                    "## 路由决策规则（严格执行：postgres 优先）\n"
                    "请根据问题特征选择合适的路由和工具：\n\n"
                    "**1. 通用健身/运动科学科普（默认推荐）**\n"
                    "   - 适用于：大多数原理、机制、概念性解释\n"
                    "   - route: local, tools: ['postgres', 'milvus']\n"
                    "   - 执行流程：postgres → 有结果则返回 → 无结果才用 milvus 兜底\n\n"
                    "**2. 明确的结构化查询（postgres 足够）**\n"
                    "   - 适用于：概念定义、简短机制说明、名词解释\n"
                    "   - route: local, tools: ['postgres']\n\n"
                    "**3. 明确需要长文本深度科普（可能需要 milvus）**\n"
                    "   - 适用于：用户明确要求“详细机制”“完整原理”“系统性解释”“长篇科普”\n\n"
                    "   - route: local, tools: ['milvus']\n\n"
                    "**4. 外部检索类（需要外网资料）**\n"
                    "   - 本地知识库可能不足，需要外部检索\n"
                    "   - route: hybrid, tools: ['milvus']\n\n"
                    "**5. 超出范围类（拒绝回答）**\n"
                    "   - 问题涉及个性化训练计划制定、疾病诊断、治疗、康复处方等内容\n"
                    "   - route: local, tools: []（空列表表示无法处理）\n\n"
                    "## 输出格式\n"
                    "请输出三个字段：\n"
                    "- route：local（本地）/ external（外部）/ hybrid（混合）\n"
                    "- tools：列表，元素为 'postgres' 和/或 'milvus'，若拒绝回答则为空列表 []\n"
                    "  - **默认推荐**: ['postgres', 'milvus'] 让系统自动优先使用 postgres\n"
                    "- rationale：中文简要说明选择理由（1-2句话）"
                ),
            ),
            (
                "human",
                "用户问题：{question}\n\n最近对话历史：\n{history}",
            ),
        ]
    )
    router_chain = router_prompt | llm.with_structured_output(KBRouteDecision)

    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是健身知识科普讲解助手，需要依据给定的检索结果进行回答。请遵循以下规则：\n"
                    "1. 仅围绕健身、运动科学与营养学的**原理性与机制性知识**进行讲解，包括训练学、运动生理学、营养代谢、器械原理、运动风险与禁忌等内容；不提供个性化训练计划、饮食方案、补剂建议或处方级指导。\n"
                    "2. 若检索结果信息不足或知识库暂无相关记载，应明确说明当前知识库中缺乏对应内容，并建议用户向其他模块或外部资源查询。\n"
                    "3. 回答应保持专业、客观、通俗易懂，使用简体中文，避免夸大效果或绝对化表述。\n"
                    "4. 如用户问题超出健身知识科普范围（如个性化指导、医疗诊断或治疗建议），应委婉拒答并简要说明原因。\n"
                    "5. 当检索结果来自不同数据源时，应区分各来源的核心观点并进行合理整合，避免简单拼接或重复叙述。\n"
                    "6. 在回答结尾列出所使用的知识来源名称或编号（如有），以增强内容的可追溯性与可信度。"
                ),
            ),
            (
                "human",
                (
                    "用户问题：{question}\n\n"
                    "Milvus 向量检索结果：\n{milvus_context}\n\n"
                    "PostgreSQL 结构化检索结果：\n{postgres_context}\n\n"
                    "外部检索结果：\n{external_context}"
                ),
            ),
        ]
    )

    async def guardrails(state: KBWorkflowState) -> Dict[str, Any]:
        """Guardrails 检查用户问题是否与知识库内容相关"""
        question = state.get("question", "")
        decision = await guardrails_chain.ainvoke({"question": question})
        summary = decision.summary or (
            "抱歉，该问题不在健身知识库的支持范围内"
            if decision.decision == "end"
            else ""
        )
        kb_logger.info("KB guardrails decision: {}", decision.decision)
        return {
            "guardrails_decision": decision.decision,
            "summary": summary,
            "steps": ["guardrails"],
        }

    def guardrails_edge(state: KBWorkflowState) -> str:
        """根据 guardrails 决策路由到最终处理或 kb_router"""
        return "finalize" if state.get("guardrails_decision") == "end" else "kb_router"

    def _history_to_text(history: List[Dict[str, str]], limit: int = 4) -> str:
        """将对话历史转换为文本格式"""
        if not history:
            return ""
        history_list: List[str] = []
        for item in history[-limit:]:
            role = item["role"]
            content = item["content"]
            history_list.append(f"{role}:{content}")
        return "\n".join(history_list)

    async def router(state: KBWorkflowState) -> Dict[str, Any]:
        """路由用户问题到合适的处理模块"""
        question = state.get("question", "")
        # 转换历史记录为文本格式
        history_text = _history_to_text(state.get("history", []))
        decision = await router_chain.ainvoke(
            {
                "question": question,
                "history": history_text,
            }
        )
        route = decision.route
        # 若外部搜索被禁用，且请求为外部路由，强制转换为本地路由
        if route in {"external", "hybrid"} and not allow_external_search:
            kb_logger.info(
                "Router requested {} but external search is disabled; using local instead.",
                route,
            )
            route = "local"
        tools = [
            tool for tool in decision.tools or [] if tool in {"milvus", "postgres"}
        ]
        # 如果路由为外部路由，且没有可用的工具，则默认使用 postgres + milvus 兜底策略
        if route != "external" and not tools:
            tools = ["postgres", "milvus"]
            kb_logger.info(
                "Router 未指定工具，使用默认策略: postgres 优先 + milvus 兜底"
            )
        kb_logger.info(
            "KB router decision: {} tools={} ({})",
            route,
            tools,
            decision.rationale,
        )
        return {
            "route": route,
            "kb_tools": tools,
            "steps": ["router"],
        }

    def router_edge(state: KBWorkflowState) -> str:
        """
        根据 router 决策路由到本地、外部或混合知识源检索
        混合检索先调用本地，再调用外部，所以先路由到 local_search，后续再路由到 external_search
        """
        return "external_search" if state.get("route") == "external" else "local_search"

    async def local_search(state: KBWorkflowState) -> Dict[str, Any]:
        """
        优先使用 PostgreSQL pgvector 结构化查询，如果无结果再用 Milvus 兜底。

        执行策略：
        1. 优先查询 PostgreSQL（如果在工具列表中）
        2. 如果 PostgreSQL 有结果（>= 1条），直接使用，跳过 Milvus
        3. 如果 PostgreSQL 无结果或未被选择，查询 Milvus 作为兜底
        """
        question = state.get("question", "")
        if not question.strip():
            return {
                "milvus_results": [],
                "postgres_results": [],
                "local_results": [],
                "steps": ["local_search"],
            }
        tools = state.get("kb_tools", ["postgres", "milvus"])

        milvus_results: List[Dict[str, Any]] = []
        postgres_results: List[Dict[str, Any]] = []
        combined_results: List[Dict[str, Any]] = []

        # Step 1: 优先查询 PostgreSQL（如果在工具列表中）
        should_try_postgres = "postgres" in tools
        should_try_milvus = "milvus" in tools

        # 进行 PostgreSQL 查询的具体逻辑
        if should_try_postgres:
            if not postgres_search_url:
                kb_logger.warning(
                    "PostgreSQL 工具被选中，但 INGEST_SERVICE_URL 未配置，跳过 PostgreSQL 直接使用 Milvus。"
                )
                should_try_milvus = True
            else:
                kb_logger.info("🔍 [优先] 查询 PostgreSQL pgvector 结构化数据库...")
                payload: Dict[str, Any] = {
                    "query": question,
                    "top_k": effective_top_k,
                }
                if settings.KB_POSTGRES_SIMILARITY_THRESHOLD is not None:
                    payload["threshold"] = settings.KB_POSTGRES_SIMILARITY_THRESHOLD
                # 正式调用 PostgreSQL 接口进行查询
                try:
                    # 设置超时时间并调用 PostgreSQL 接口(http post)
                    timeout_cfg = aiohttp.ClientTimeout(total=request_timeout)
                    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                        async with session.post(
                            postgres_search_url, json=payload
                        ) as response:
                            # 响应状态码为 200 的操作
                            if response.status == 200:
                                body = await response.json()
                                data_results = body.get("results") or []
                                if isinstance(data_results, list):
                                    for idx, item in enumerate(data_results):
                                        item_copy = dict(item)
                                        item_copy["metadata"] = dict(
                                            item_copy.get("metadata", {})
                                        )
                                        item_copy["tool"] = "postgres"

                                        # 处理相似度分数
                                        similarity = item_copy.get(
                                            "similarity"
                                        ) or item_copy.get("score")
                                        try:
                                            item_copy["similarity"] = (
                                                float(similarity)
                                                if similarity is not None
                                                else 0.0
                                            )
                                        except (TypeError, ValueError):
                                            item_copy["similarity"] = 0.0

                                        # 处理文档ID
                                        doc_id = None
                                        for key in ["id", "document_id", "source_id"]:
                                            if item_copy.get(key):
                                                doc_id = item_copy.get(key)
                                                break
                                        item_copy["id"] = str(
                                            doc_id or f"postgres_{idx}"
                                        )
                                        postgres_results.append(item_copy)
                                    # 重排序
                                    if (
                                        postgres_results
                                        and knowledge_service.enable_rerank
                                    ):
                                        postgres_results = (
                                            await knowledge_service.rerank_candidates(
                                                question,
                                                postgres_results,
                                                top_k=effective_top_k,
                                            )
                                        )
                                    filtered_postgres: List[Dict[str, Any]] = []
                                    # 根据分数的阈值，过滤 PostgreSQL 结果
                                    for doc in postgres_results:
                                        similarity = float(
                                            doc.get("similarity")
                                            or doc.get("score")
                                            or 0.0
                                        )
                                        rerank_score = float(
                                            doc.get("rerank_score", 0.0)
                                        )
                                        if knowledge_service.enable_rerank:
                                            if (
                                                similarity
                                                >= settings.KB_POSTGRES_SIMILARITY_THRESHOLD
                                                and rerank_score
                                                >= settings.KB_POSTGRES_RERANK_SCORE_THRESHOLD
                                            ):
                                                filtered_postgres.append(doc)
                                        else:
                                            if (
                                                similarity
                                                >= settings.KB_POSTGRES_SIMILARITY_THRESHOLD
                                            ):
                                                filtered_postgres.append(doc)
                                    # 截断结果,返回 top_k 条
                                    postgres_results = filtered_postgres[
                                        :effective_top_k
                                    ]
                                    kb_logger.info(
                                        "✅ PostgreSQL 返回 {} 条结果，过滤后保留 {} 条",
                                        len(data_results),
                                        len(postgres_results),
                                    )
                                else:
                                    kb_logger.warning(
                                        "Unexpected PostgreSQL search payload structure: {}",
                                        body,
                                    )
                            # 处理非 200 响应
                            else:
                                error_text = await response.text()
                                kb_logger.warning(
                                    "PostgreSQL KB search failed ({}): {}",
                                    response.status,
                                    error_text,
                                )
                except Exception as e:
                    kb_logger.error("PostgreSQL knowledge search error: {}", e)

        # Step 2: 根据 PostgreSQL 结果决定是否需要 Milvus 兜底
        if postgres_results and len(postgres_results) >= 1:
            kb_logger.info(
                "✅ PostgreSQL 有结果（{}条），直接使用结构化数据，跳过 Milvus 向量查询",
                len(postgres_results),
            )
            combined_results = postgres_results
        # 没有 PostgreSQL 结果，执行 Milvus 向量查询
        else:
            if should_try_milvus:
                kb_logger.info("⚠️ PostgreSQL 无结果，使用 Milvus 向量库兜底...")
                try:
                    # 执行 Milvus 向量查询
                    docs = await knowledge_service.search(
                        query=question,
                        top_k=effective_top_k,
                        similarity_threshold=settings.KB_MILVUS_SIMILARITY_THRESHOLD,
                        filter_expr=filter_expr,
                        filter_by_similarity=not knowledge_service.enable_rerank,
                    )
                    # 通过doc.copy对象，重新整理一下字典里的各种字段，然后加到 milvus_results 里
                    for doc in docs:
                        doc.copy = dict(doc)
                        doc.copy["metadata"] = dict(doc.get("metadata", {}))
                        doc.copy["tool"] = "milvus"
                        milvus_results.append(doc.copy)
                    kb_logger.info(
                        "✅ Milvus 返回 {} 条结果",
                        len(milvus_results),
                    )
                    combined_results = milvus_results
                except Exception as e:
                    kb_logger.error("Milvus knowledge search error: {}", e)
                    combined_results = []
            else:
                kb_logger.warning(
                    "⚠️ PostgreSQL 无结果，且未分配 Milvus 向量查询工具，直接返回空结果"
                )
                combined_results = []

        return {
            "milvus_results": milvus_results,
            "postgres_results": postgres_results,
            "combined_results": combined_results,
            "route": state.get("route", "local"),
            "steps": ["local_search"],
        }

    def local_edge(state: KBWorkflowState) -> str:
        """本地知识库路由决策"""
        route = state.get("route", "local")
        if route in {"hybrid", "external"} and allow_external_search and external_url:
            return "external_search"
        return "finalize"

    async def external_search(state: KBWorkflowState) -> Dict[str, Any]:
        """外部搜索"""
        if not (allow_external_search and external_url):
            kb_logger.warning(
                "Skip external search: 外部知识库路由未配置，无法进行外部搜索"
            )
            return {"external_results": [], "steps": ["external_search"]}

        if external_is_postgres and "postgres" in state.get("kb_tools", []):
            kb_logger.debug(
                "Skip external search: router already执行了 PostgreSQL 工具，且外部检索与其同源。"
            )
            return {"external_results": [], "steps": ["external_search"]}

        question = state.get("question", "")
        if not question.strip():
            kb_logger.warning("Skip external search: 问题为空，无法进行外部搜索")
            return {"external_results": [], "steps": ["external_search"]}

        payload: Dict[str, Any] = {
            "query": question,
            "top_k": effective_top_k,
        }
        if effective_threshold:
            payload["threshold"] = effective_threshold

        external_results: List[Dict[str, Any]] = []
        try:
            timeout_cfg = aiohttp.ClientTimeout(total=request_timeout)
            async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                async with session.post(external_url, json=payload) as response:
                    # 响应成功
                    if response.status == 200:
                        body = await response.json()
                        data_results = body.get("results", [])
                        if isinstance(data_results, list):
                            external_results = data_results
                        else:
                            kb_logger.warning(
                                "Skip external search: 外部搜索返回结果格式错误，无法解析。"
                            )
                    # 响应失败
                    else:
                        error_text = await response.text()
                        kb_logger.error(
                            "External search failed with status {}: {}",
                            response.status,
                            error_text,
                        )
        except Exception as e:
            kb_logger.error(f"External search error: {e}", e)

        return {"external_results": external_results, "steps": ["external_search"]}

    def _format_results(
        results: List[Dict[str, Any]],
        *,
        default_label: str,
        empty_hint: str,
    ) -> str:
        if not results:
            return empty_hint

        formatted_results: List[str] = []
        for idx, doc in enumerate(results[:effective_top_k]):
            content = doc.get("content", "").strip()
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "")
            tool_label = default_label
            tag = f"[{tool_label}#{idx + 1}]{content}"
            if source:
                tag = f"{tag}\n来源:{source}"
            formatted_results.append(tag)
        return "\n\n".join(formatted_results)

    def _format_milvus_results(results: List[Dict[str, Any]]) -> str:
        return _format_results(
            results, default_label="Milvus", empty_hint="⚠️ Milvus 无结果"
        )

    def _format_postgres_results(results: List[Dict[str, Any]]) -> str:
        return _format_results(
            results, default_label="Postgres", empty_hint="⚠️ Postgres 无结果"
        )

    def _format_local_results(results: List[Dict[str, Any]]) -> str:
        return _format_results(
            results, default_label="本地知识库", empty_hint="⚠️ 本地知识库无结果"
        )

    def _format_external_results(results: List[Dict[str, Any]]) -> str:
        return _format_results(
            results, default_label="外部知识库", empty_hint="⚠️ 外部知识库无结果"
        )

    def _collect_sources(*result_sets: List[Dict[str, Any]]) -> List[str]:
        """收集所有结果集中的来源
        注意：result_sets是一个元组，元组中的每个元素都是一个List[Dict[str, Any]]
        """
        collected: List[str] = []
        for dataset in result_sets:
            for doc in dataset:
                meta = doc.get("metadata", {})
                candidate = meta.get("source", "")
                if candidate:
                    collected.append(str(candidate))
        # 利用字典的键值对唯一性，自动去重
        # 注意：这里不能用set，因为set是无序的，而我们需要保持结果集的顺序
        seen: Dict[str, None] = {}
        for source in collected:
            seen.setdefault(source, None)
        return list(seen.keys())

    async def finalize(state: KBWorkflowState) -> KBOutputState:
        """最终回答"""
        if state.get("guardrails_decision") == "end":
            summary = state.get("summary", "抱歉，该问题暂时无法回答。")
            return {"answer": summary, "steps": ["finalize"], "sources": []}

        milvus_results = state.get("milvus_results", [])
        postgres_results = state.get("postgres_results", [])
        local_results = state.get("local_results", []) or (
            milvus_results + postgres_results
        )
        external_results = state.get("external_results", [])

        milvus_context = _format_milvus_results(milvus_results)
        postgres_context = _format_postgres_results(postgres_results)
        local_context = _format_local_results(local_results)
        external_context = _format_external_results(external_results)

        sources = _collect_sources(milvus_results, postgres_results, external_results)

        prompt = final_prompt.format_messages(
            question=state.get("question", ""),
            milvus_context=milvus_context,
            postgres_context=postgres_context,
            external_context=external_context,
        )

        try:
            response = await llm.ainvoke(prompt)
            content = getattr(response, "content", None)
            if isinstance(content, str):
                answer = content.strip()
            else:
                answer = str(content).strip()
        except Exception as e:
            kb_logger.error(f"Finalize error: {e}", e)

        if not answer:
            answer = "检索已完成，但当前无法生成可靠的菜谱文化回答。"

        return {
            "answer": answer,
            "steps": ["finalize"],
            "sources": sources,
        }

    # 定义状态图
    graph_builder = StateGraph(
        KBWorkflowState,
        input=KBInputState,
        output=KBOutputState,
    )

    graph_builder.add_node("guardrails", guardrails)
    graph_builder.add_node("kb_router", router)
    graph_builder.add_node("local_search", local_search)
    graph_builder.add_node("external_search", external_search)
    graph_builder.add_node("finalize", finalize)

    graph_builder.add_edge(START, "guardrails")
    graph_builder.add_conditional_edges("guardrails", guardrails_edge)
    graph_builder.add_conditional_edges("kb_router", router_edge)
    graph_builder.add_conditional_edges("local_search", local_edge)
    graph_builder.add_edge("external_search", "finalize")
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile()
