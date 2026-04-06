from langchain_neo4j import Neo4jGraph
from loguru import logger
import os

# 配置 Neo4j 相关日志级别（通过环境变量）
os.environ["NEO4J_MODULE_LOG_LEVEL"] = "ERROR"

# 尝试从 settings 获取配置，如果不可用则使用默认值
try:
    from muscleking.config import settings

    NEO4J_URI = getattr(settings, "NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = getattr(settings, "NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = getattr(settings, "NEO4J_PASSWORD", "muscleking")
    NEO4J_DATABASE = getattr(settings, "NEO4J_DATABASE", "neo4j")
except ImportError:
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "muscleking"
    NEO4J_DATABASE = "neo4j"

# 获取日志记录器
log = logger.bind(service="neo4jconn")


def get_neo4j_graph() -> Neo4jGraph:
    """
    创建并返回一个Neo4jGraph实例，使用配置文件中的设置。

    Returns:
        Neo4jGraph: 配置好的Neo4j图数据库连接实例
    """
    log.info(f"initialize Neo4j connection: {NEO4J_URI}")

    try:
        kwargs = {
            "url": NEO4J_URI,
            "database": NEO4J_DATABASE,
        }
        if NEO4J_USER and NEO4J_PASSWORD not in (None, ""):
            kwargs.update(
                {
                    "username": NEO4J_USER,
                    "password": NEO4J_PASSWORD,
                }
            )

        neo4j_graph = Neo4jGraph(**kwargs)
        return neo4j_graph
    except Exception as e:
        log.error(f"Failed to connect to Neo4j: {e}")
        raise
