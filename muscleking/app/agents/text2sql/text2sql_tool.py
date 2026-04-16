"""
Text2SQL tool wrapper for the multi-tool workflow.

This node bridges the planner/tool-selection workflow with the
Text2SQL LangGraph pipeline implemented under ``gustobot.application.agents.text2sql``.
"""

from typing import Dict, Any, Coroutine, Callable, List
from muscleking.app.storage.core.neo4jconn import get_neo4j_graph
from muscleking.app.agents.text2sql.text2sql_workflow import create_text2sql_workflow
from loguru import logger
from muscleking.app.agents.cyper_tools.cypher_node import CypherQueryOutputState
from langchain_openai import ChatOpenAI
from muscleking.app.config.settings import settings


logger = logger.bind(service="text2sql")


def create_text2sql_tool_node(
    neo4j_graph=None,
) -> Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Create a LangGraph node that executes the Text2SQL workflow.

    Parameters
    ----------
    neo4j_graph : Neo4jGraph | None
        Existing Neo4j graph connection used for schema retrieval. If ``None``,
        the node will attempt to obtain one via ``get_neo4j_graph``.
    """

    async def text2sql_query(state: Dict[str, Any]) -> Dict[str, Any]:
        question = state.get("task") or state.get("question") or ""
        tool_args: Dict[str, Any] = state.get("query_parameters", {}) or {}

        connection_id = tool_args.get("connection_id")
        db_type = tool_args.get("db_type") or "MySQL"
        max_rows = int(tool_args.get("max_rows") or 1000)
        connection_string = tool_args.get("connection_string")
        max_retries = int(tool_args.get("max_retries") or 3)

        errors: List[str] = []

        graph = neo4j_graph
        if graph is None and get_neo4j_graph is not None:
            try:
                graph = get_neo4j_graph()
                logger.info("Obtained Neo4j graph connection for Text2SQL tool.")
            except Exception as e:
                logger.error("Failed to obtain Neo4j graph connection: %s", e)
                errors.append(f"无法连接图数据库: {e}")
                graph = None

        # neo4j链接失败兜底机制

        # 若LLMAPI无效，兜底机制

        # 大模型初始化

        text2sql_llm = ChatOpenAI(
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            base_url=settings.LLM_BASE_URL,
            temperature=0.0,
            tags=["text2sql"],
        )
        # workflow搭建
        workflow = create_text2sql_workflow(
            llm=text2sql_llm,
            neo4j_graph=graph,
            db_type=db_type,
            connection_string=connection_string,
            max_retries=max_retries,
        )

        # input_state定义
        input_state = {
            "question": question,
            "connection_id": connection_id,
            "db_type": db_type,
            "max_retries": max_retries,
            "max_rows": max_rows,
        }
        # workflow执行
        try:
            result: Dict[str, Any] = await workflow.ainvoke(input_state)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Text2SQL workflow execution failed: %s", exc)
            error_message = f"执行数据库查询失败：{exc}"
            errors.append(error_message)
            answer = "抱歉，执行数据库查询时发生异常，请稍后再试。"
            records_payload = {
                "answer": answer,
                "rows": [],
                "sql": "",
                "error": error_message,
            }

        answer = result.get("answer") or ""
        sql_statement = result.get("sql_statement") or ""
        execution_results = result.get("execution_results") or []
        execution_error = result.get("execution_error") or []
        visualization = result.get("visualization")
        viz_config = result.get("visualization_config")

        records_payload = {
            "answer": answer,
            "rows": execution_results,
            "sql": sql_statement,
            "visualization": visualization,
        }
        if viz_config:
            records_payload["visualization_config"] = viz_config
        # 查询结果映射与payload(负载)构造
        return {
            "cyphers": [
                CypherQueryOutputState(
                    **{
                        "task": question,
                        "query": sql_statement,
                        "errors": errors,
                        "records": records_payload,
                        "steps": ["execute_text2sql_query"],
                    }
                )
            ],
            "steps": ["execute_text2sql_query"],
        }

    return text2sql_query
