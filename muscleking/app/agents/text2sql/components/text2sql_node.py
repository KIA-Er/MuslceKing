from __future__ import annotations

import json
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Set, Iterable
from loguru import logger
from langchain_core.language_models import BaseChatModel
from muscleking.app.agents.text2sql.components.text2sql_prompt import (
    format_schema_as_text,
    create_query_analysis_prompt,
    create_sql_generation_prompt,
    create_visualization_prompt,
)
from muscleking.app.agents.text2sql.components.models import (
    SQLAnalysis,
    VisualizationRecommendation,
)
from muscleking.app.agents.text2sql.components.utils import render_analysis_markdown
from langchain_core.output_parsers import StrOutputParser
from .validators import validate_sql_syntax, validate_sql_security
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from muscleking.app.config import settings
from muscleking.app.agents.text2sql.components.prompt import (
    COLUMN_DESCRIPTIONS,
    TABLE_DESCRIPTIONS,
    RELATIONSHIP_FACTS,
)
import asyncio
import re

logger = logger.bind(service="text2sql_node")


def create_answer_formatter_node() -> Callable[
    [Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]
]:
    """Build a LangGraph node that composes the final answer for the user."""

    async def format_answer(state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("-----格式化最终回答-----")

        execution_error = state.get("execution_error")
        sql_statement = state.get("sql_statement", "")
        results = state.get("execution_results") or []
        analysis_text = state.get("analysis_text") or ""
        visualization = state.get("visualization") or {}
        question = state.get("question", "")

        if execution_error:
            answer = f"抱歉，执行 SQL 时出现错误：{execution_error}"
        else:
            answer_lines: List[str] = []
            answer_lines.append(f"### 查询结果摘要\n问题：{question or '（未提供）'}")
            if analysis_text:
                answer_lines.append("\n---\n")
                answer_lines.append(analysis_text)
            if results:
                answer_lines.append("\n---\n### 结果预览")
                preview = json.dumps(results[:5], ensure_ascii=False, indent=2)
                answer_lines.append(f"```json\n{preview}\n```")
            if visualization:
                answer_lines.append("\n---\n### 可视化建议")
                answer_lines.append(
                    f"- 类型：{visualization.get('chart_type', 'table')}\n"
                    f"- 标题：{visualization.get('title', '查询结果')}"
                )
                config = (
                    visualization.get("config")
                    if isinstance(visualization, dict)
                    else None
                )
                if config:
                    answer_lines.append("```json")
                    answer_lines.append(
                        json.dumps(config, ensure_ascii=False, indent=2)
                    )
                    answer_lines.append("```")
            answer = "\n".join(answer_lines).strip()

        viz_config = None
        if isinstance(visualization, dict):
            viz_config = visualization.get("config")

        return {
            "answer": answer,
            "sql_statement": sql_statement,
            "execution_results": results,
            "visualization": visualization if isinstance(visualization, dict) else None,
            "visualization_config": viz_config,
            "steps": ["format_answer"],
        }

    return format_answer


"""
Query analysis node.
"""


def create_query_analysis_node(
    llm: BaseChatModel,
) -> Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]:
    """Build a LangGraph node that performs structured query analysis."""

    prompt = create_query_analysis_prompt()
    analysis_chain = prompt | llm.with_structured_output(SQLAnalysis)

    async def analyze(state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("-----开始分析查询意图-----")

        question = state.get("question", "")
        schema_context = state.get("schema_context") or {}
        # value_mappings 功能已简化移除
        mappings_str = ""

        schema_text = format_schema_as_text(schema_context)

        inputs = {
            "db_type": state.get("db_type", "MySQL"),
            "schema": schema_text,
            "value_mappings": mappings_str or "无值映射信息",
            "question": question,
        }

        analysis: SQLAnalysis = await analysis_chain.ainvoke(inputs)
        analysis_dict = analysis.model_dump()
        rendered_markdown = render_analysis_markdown(analysis, None)

        logger.info(
            "查询分析完成，涉及表：%s", ", ".join(analysis.required_tables or [])
        )

        return {
            "analysis": analysis_dict,
            "analysis_text": rendered_markdown,
            "steps": ["query_analysis"],
        }

    return analyze


"""
SQL generation node.
"""


def create_sql_generation_node(
    llm: BaseChatModel,
    db_type: str = "MySQL",
) -> Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Build a LangGraph node that generates SQL statements from the analysed query.
    """
    prompt = create_sql_generation_prompt()
    sql_llm = llm.with_config(temperature=0.1)
    sql_chain = prompt | sql_llm | StrOutputParser()

    async def generate_sql(state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("-----开始生成 SQL 语句-----")

        schema_context = state.get("schema_context") or {}
        analysis_dict = state.get("analysis")
        analysis_text = state.get("analysis_text") or ""

        if not schema_context:
            logger.warning("缺少 Schema 信息，无法生成 SQL")
            return {
                "sql_statement": "",
                "steps": ["sql_generation_failed_no_schema"],
            }

        schema_text = format_schema_as_text(schema_context)
        # value_mappings 功能已简化移除
        mappings_str = ""
        analysis_summary = analysis_text or render_analysis_markdown(analysis_dict, "")

        question = state.get("question", "")
        inputs = {
            "db_type": db_type,
            "schema": schema_text,
            "value_mappings": mappings_str or "无值映射信息",
            "analysis_summary": analysis_summary or "无分析信息",
            "question": question,
        }

        try:
            sql_raw = await sql_chain.ainvoke(inputs)
            sql_statement = _clean_sql_statement(sql_raw)
            logger.info("SQL 生成完成")
            return {
                "sql_statement": sql_statement,
                "steps": ["sql_generation"],
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("SQL 生成失败: %s", exc)
            return {
                "sql_statement": "",
                "steps": ["sql_generation_failed"],
            }

    return generate_sql


def _clean_sql_statement(sql: str) -> str:
    """
    Remove Markdown code fences and normalise whitespace.
    """
    cleaned = sql.replace("```sql", "").replace("```", "").strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines)


"""
SQL validation node.
"""


def create_sql_validation_node(
    db_type: str = "MySQL",
) -> Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Build a LangGraph node that validates SQL produced by upstream generators.
    """

    async def validate_sql(state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("-----开始验证 SQL 语句-----")

        sql_statement = state.get("sql_statement", "")
        current_retry = state.get("retry_count", 0)

        if not sql_statement:
            logger.warning("SQL 语句为空，跳过验证")
            return {
                "is_valid": False,
                "validation_errors": ["SQL 语句为空"],
                "retry_count": current_retry + 1,
                "steps": ["sql_validation_failed"],
            }

        errors: List[str] = []

        try:
            syntax_ok, syntax_errors = validate_sql_syntax(sql_statement, db_type)
            if not syntax_ok:
                errors.extend(syntax_errors)
                logger.warning("SQL 语法验证失败: %s", syntax_errors)

            security_ok, security_warnings = validate_sql_security(sql_statement)
            if not security_ok:
                errors.extend(security_warnings)
                logger.warning("SQL 安全检查警示: %s", security_warnings)

            is_valid = not errors
            next_retry = current_retry if is_valid else current_retry + 1

            if is_valid:
                logger.info("SQL 验证通过")
            else:
                logger.error("SQL 验证失败: %s", errors)

            return {
                "is_valid": is_valid,
                "validation_errors": errors,
                "retry_count": next_retry,
                "steps": ["sql_validation" if is_valid else "sql_validation_failed"],
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("SQL 验证过程出错: %s", exc)
            return {
                "is_valid": False,
                "validation_errors": [f"验证过程出错: {exc}"],
                "retry_count": current_retry + 1,
                "steps": ["sql_validation_error"],
            }

    return validate_sql


"""
SQL execution node.
"""


def create_sql_execution_node(
    connection_string: Optional[str] = None,
) -> Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Build a LangGraph node that executes read-only SQL statements.
    """

    async def execute_sql(state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("-----开始执行 SQL 查询-----")

        sql_statement = (state.get("sql_statement") or "").strip()
        connection_id = state.get("connection_id")
        max_rows = int(state.get("max_rows") or 1000)

        if not sql_statement:
            logger.warning("SQL 语句为空，跳过执行")
            return {
                "execution_results": None,
                "execution_error": "SQL 语句为空",
                "steps": ["sql_execution_skipped"],
            }

        if not _is_read_only_query(sql_statement):
            logger.warning("检测到非只读查询，已阻止执行")
            return {
                "execution_results": None,
                "execution_error": "仅支持只读 SELECT 查询",
                "steps": ["sql_execution_blocked"],
            }

        if not state.get("is_valid"):
            logger.warning("SQL 验证未通过，跳过执行")
            return {
                "execution_results": None,
                "execution_error": "SQL 验证未通过",
                "steps": ["sql_execution_skipped_invalid"],
            }

        conn_str = connection_string or _get_connection_string(connection_id)
        if not conn_str:
            logger.error("无法获取数据库连接信息 connection_id=%s", connection_id)
            return {
                "execution_results": None,
                "execution_error": "无法获取数据库连接信息",
                "steps": ["sql_execution_failed"],
            }

        try:
            results = await _execute_sql_query(
                conn_str, sql_statement, max_rows=max_rows
            )
            logger.info("SQL 执行成功，返回 %d 行结果", len(results))
            return {
                "execution_results": results,
                "execution_error": None,
                "steps": ["sql_execution"],
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("SQL 执行失败: %s", exc)
            return {
                "execution_results": None,
                "execution_error": str(exc),
                "steps": ["sql_execution_failed"],
            }

    return execute_sql


def _map_db_type_to_driver(db_type: str) -> str:
    mapping = {
        "mysql": "mysql+pymysql",
        "mariadb": "mysql+pymysql",
        "postgresql": "postgresql+psycopg2",
        "postgres": "postgresql+psycopg2",
        "pg": "postgresql+psycopg2",
        "sqlite": "sqlite",
    }
    return mapping.get((db_type or "").lower(), db_type)


def _get_connection_string(connection_id: Optional[int]) -> Optional[str]:
    """
    直接返回 MySQL 连接字符串，不再支持 SQLite 和 dbconnection 表查询。
    简化实现，统一使用 MySQL 作为 Text2SQL 的目标数据库。
    """
    logger.debug("使用默认 MySQL 数据库 URL: %s", settings.DATABASE_URL)
    return settings.DATABASE_URL


async def _execute_sql_query(
    connection_string: str,
    sql: str,
    max_rows: int = 1000,
) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(_run_query_sync, connection_string, sql, max_rows)


def _run_query_sync(
    connection_string: str,
    sql: str,
    max_rows: int,
) -> List[Dict[str, Any]]:
    engine: Engine = create_engine(connection_string, future=True)
    try:
        with engine.connect() as connection:
            result = connection.execution_options(stream_results=True).execute(
                text(sql)
            )
            columns = list(result.keys())
            rows = result.fetchmany(max_rows)
            return [
                {column: row[idx] for idx, column in enumerate(columns)} for row in rows
            ]
    finally:
        engine.dispose()


def _is_read_only_query(sql: str) -> bool:
    """
    检查 SQL 是否为只读查询（SELECT、WITH、EXPLAIN）

    Returns:
        True if query is read-only, False otherwise
    """
    statement = sql.strip()
    if not statement:
        return False

    # 检查是否包含多条语句（;分隔）
    if ";" in statement.rstrip(";"):
        return False

    upper = statement.upper()

    # 允许的只读关键词
    readonly_keywords = ["SELECT ", "WITH ", "EXPLAIN ", "SHOW "]
    starts_with_readonly = any(upper.startswith(kw) for kw in readonly_keywords)

    # 危险关键词（即使在SELECT中也不允许）
    dangerous_keywords = [
        "INSERT ",
        "UPDATE ",
        "DELETE ",
        "DROP ",
        "CREATE ",
        "ALTER ",
        "TRUNCATE ",
        "REPLACE ",
        "MERGE ",
        "GRANT ",
        "REVOKE ",
        "EXEC ",
        "EXECUTE ",
        "CALL ",
    ]
    contains_dangerous = any(kw in upper for kw in dangerous_keywords)

    return starts_with_readonly and not contains_dangerous


"""
Visualization recommendation node.
"""


def create_visualization_node(
    llm: BaseChatModel,
) -> Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Build a LangGraph node that recommends a visualization configuration based
    on the SQL query results.
    """
    prompt = create_visualization_prompt()
    viz_llm = llm.with_structured_output(VisualizationRecommendation)

    async def recommend(state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("-----开始生成可视化建议-----")

        results: List[Dict[str, Any]] = state.get("execution_results") or []
        if not results:
            logger.info("查询结果为空，默认返回表格展示")
            return {
                "visualization": VisualizationRecommendation(
                    chart_type="table",
                    title="查询结果",
                ).model_dump(),
                "steps": ["visualization_default"],
            }

        sample_rows = json.dumps(results[:10], ensure_ascii=False, indent=2)
        question = state.get("question", "")
        analysis_summary = state.get("analysis_text") or ""
        sql_statement = state.get("sql_statement") or ""

        schema_text = format_schema_as_text(state.get("schema_context") or {})

        inputs = {
            "question": question,
            "analysis_summary": analysis_summary or schema_text,
            "sql_statement": sql_statement,
            "sample_rows": sample_rows,
        }

        try:
            recommendation = await viz_llm.ainvoke(prompt.format_messages(**inputs))
            logger.info("可视化推荐完成: %s", recommendation.chart_type)
            return {
                "visualization": recommendation.model_dump(),
                "steps": ["visualization"],
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("可视化推荐失败: %s", exc)
            return {
                "visualization": VisualizationRecommendation(
                    chart_type="table",
                    title="查询结果",
                ).model_dump(),
                "steps": ["visualization_failed"],
            }

    return recommend


"""
Schema retrieval node.

Fetches tables, columns, and relationships directly from MySQL INFORMATION_SCHEMA.
简化实现，不再依赖 Neo4j，直接从 MySQL 数据库读取表结构。
"""

_STOP_WORDS: Set[str] = {
    "the",
    "and",
    "for",
    "from",
    "with",
    "into",
    "what",
    "which",
    "when",
    "who",
    "where",
    "how",
    "many",
    "much",
    "that",
    "this",
    "those",
    "these",
    "all",
    "any",
    "year",
    "month",
    "day",
    "query",
    "please",
    "show",
    "list",
    "give",
    "查询",
    "一下",
    "所有",
    "哪些",
    "什么",
    "数据",
}


def create_schema_retrieval_node(
    neo4j_graph=None,  # 保留参数以兼容旧代码，但不再使用
) -> Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Build a LangGraph node that enriches the workflow state with schema
    metadata from MySQL INFORMATION_SCHEMA.
    """

    async def retrieve_schema(state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("-----开始检索数据库 Schema (直接从 MySQL)-----")

        question = (state.get("question") or "").strip()

        try:
            schema_context = await _retrieve_schema_from_mysql(question)

            logger.info(
                "Schema 检索完成，匹配到 %d 张表", len(schema_context.get("tables", []))
            )

            return {
                "schema_context": schema_context,
                "value_mappings": {},  # 简化实现，不再使用值映射
                "mappings_str": "",
                "steps": ["schema_retrieval"],
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Schema 检索失败: %s", exc)
            return {
                "schema_context": {},
                "value_mappings": {},
                "mappings_str": "",
                "steps": ["schema_retrieval_failed"],
            }

    return retrieve_schema


def _extract_keywords(question: str) -> List[str]:
    if not question:
        return []

    tokens = re.findall(r"[a-zA-Z0-9_]+", question.lower())
    return [token for token in tokens if token and token not in _STOP_WORDS]


def _score_table(table: Dict[str, Any], keywords: Iterable[str]) -> float:
    if not keywords:
        return 0.0

    score = 0.0
    name = (table.get("table_name") or "").lower()
    description = (table.get("description") or "").lower()

    for kw in keywords:
        if kw in name:
            score += 2.0
        if kw in description:
            score += 1.0

    for column in table.get("columns", []):
        column_name = (column.get("column_name") or "").lower()
        column_desc = (column.get("description") or "").lower()
        for kw in keywords:
            if kw in column_name:
                score += 1.5
            if kw in column_desc:
                score += 0.75

    return score


async def _retrieve_schema_from_mysql(question: str) -> Dict[str, Any]:
    """
    直接从 MySQL INFORMATION_SCHEMA 读取表结构信息。
    """
    import asyncio

    def _sync_retrieve():
        engine: Engine = create_engine(settings.DATABASE_URL, future=True)
        try:
            with engine.connect() as conn:
                # 获取所有表信息
                tables_query = text("""
                    SELECT
                        TABLE_NAME,
                        TABLE_COMMENT
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_TYPE = 'BASE TABLE'
                    ORDER BY TABLE_NAME
                """)
                table_rows = conn.execute(tables_query).fetchall()

                tables: List[Dict[str, Any]] = []

                for table_row in table_rows:
                    table_name = table_row[0]
                    table_comment = table_row[1] or ""
                    table_key = table_name.lower()

                    # 使用领域知识库补充描述
                    table_description = (
                        table_comment or TABLE_DESCRIPTIONS.get(table_key) or ""
                    )

                    # 获取列信息
                    columns_query = text("""
                        SELECT
                            COLUMN_NAME,
                            DATA_TYPE,
                            COLUMN_COMMENT,
                            COLUMN_KEY
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = DATABASE()
                        AND TABLE_NAME = :table_name
                        ORDER BY ORDINAL_POSITION
                    """)
                    column_rows = conn.execute(
                        columns_query, {"table_name": table_name}
                    ).fetchall()

                    columns: List[Dict[str, Any]] = []
                    for col_row in column_rows:
                        column_name = col_row[0]
                        data_type = col_row[1]
                        column_comment = col_row[2] or ""
                        column_key = col_row[3]

                        # 使用领域知识库补充描述
                        column_desc_key = (table_key, column_name.lower())
                        column_description = (
                            column_comment
                            or COLUMN_DESCRIPTIONS.get(column_desc_key)
                            or ""
                        )

                        columns.append(
                            {
                                "column_name": column_name,
                                "data_type": data_type,
                                "description": column_description,
                                "is_primary_key": column_key == "PRI",
                                "is_foreign_key": column_key in ("MUL", "FK"),
                                "is_unique": column_key == "UNI",
                            }
                        )

                    tables.append(
                        {
                            "table_name": table_name,
                            "description": table_description,
                            "columns": columns,
                        }
                    )

                # 获取外键关系
                fk_query = text("""
                    SELECT
                        kcu.TABLE_NAME as source_table,
                        kcu.COLUMN_NAME as source_column,
                        kcu.REFERENCED_TABLE_NAME as target_table,
                        kcu.REFERENCED_COLUMN_NAME as target_column
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                    WHERE kcu.TABLE_SCHEMA = DATABASE()
                    AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
                    ORDER BY kcu.TABLE_NAME, kcu.COLUMN_NAME
                """)
                fk_rows = conn.execute(fk_query).fetchall()

                relationships: List[Dict[str, Any]] = []
                seen_keys: Set[Tuple[str, str, str, str]] = set()

                for fk_row in fk_rows:
                    key = (fk_row[0], fk_row[1], fk_row[2], fk_row[3])
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    relationships.append(
                        {
                            "source_table": fk_row[0],
                            "source_column": fk_row[1],
                            "target_table": fk_row[2],
                            "target_column": fk_row[3],
                            "relationship_type": "FOREIGN_KEY",
                            "description": f"{fk_row[0]}.{fk_row[1]} references {fk_row[2]}.{fk_row[3]}",
                        }
                    )

                # 添加领域知识关系
                table_name_lookup = {
                    t["table_name"].lower(): t["table_name"] for t in tables
                }
                for relationship in RELATIONSHIP_FACTS:
                    source_lower = relationship["source_table"].lower()
                    target_lower = relationship["target_table"].lower()
                    if (
                        source_lower not in table_name_lookup
                        or target_lower not in table_name_lookup
                    ):
                        continue

                    source_name = table_name_lookup[source_lower]
                    target_name = table_name_lookup[target_lower]

                    key = (
                        source_name,
                        relationship["source_column"],
                        target_name,
                        relationship["target_column"],
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    relationships.append(
                        {
                            "source_table": source_name,
                            "source_column": relationship["source_column"],
                            "target_table": target_name,
                            "target_column": relationship["target_column"],
                            "relationship_type": relationship.get(
                                "relationship_type", ""
                            ),
                            "description": relationship.get("description", ""),
                        }
                    )

                return tables, relationships
        finally:
            engine.dispose()

    # 在线程池中执行同步查询
    tables, relationships = await asyncio.to_thread(_sync_retrieve)

    # 根据问题关键词过滤表
    keywords = _extract_keywords(question)
    if keywords:
        scored_tables = [(table, _score_table(table, keywords)) for table in tables]
        positive = [item for item in scored_tables if item[1] > 0]
        if positive:
            positive.sort(key=lambda item: item[1], reverse=True)
            max_tables = 6
            tables = [item[0] for item in positive[:max_tables]]

            # 过滤关系，只保留相关表的关系
            table_names = {table["table_name"] for table in tables}
            relationships = [
                rel
                for rel in relationships
                if rel["source_table"] in table_names
                and rel["target_table"] in table_names
            ]

    schema_context = {
        "tables": tables,
        "relationships": relationships,
    }
    return schema_context
