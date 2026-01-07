"""
MuscleKing 配置管理模块.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """应用配置"""

    # Application metadata
    APP_NAME: str = "MuscleKing"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # API configuration
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # LLM configuration
    LLM_PROVIDER: str = Field(default="openai", description="LLM provider")
    LLM_MODEL: str = Field(default="gpt-4o-mini", description="LLM model name")
    LLM_API_KEY: Optional[str] = Field(default=None, description="LLM_API_KEY")
    LLM_BASE_URL: Optional[str] = Field(default=None, description="LLM API base URL")
    
    # 兼容性配置 - 为了兼容lg_builder.py中的配置名称
    # 这些字段在__init__中动态设置

    # CORS配置 - 使用字符串，在代码中分割
    # 允许以下端口访问后端api
    CORS_ORIGINS: str = Field(
        default="http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000",
        description="Allowed CORS origins",
    )

    # 数据库配置
    DATABASE_URL: str = Field(
        default="mysql+pymysql://muscleking_user:musclepass@localhost:3306/muscleking_db",
        description="Database connection URL"
    )

    # Neo4j 图数据库配置
    NEO4J_URI: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    NEO4J_USER: str = Field(default="neo4j", description="Neo4j username")
    NEO4J_PASSWORD: str = Field(default="muscleking", description="Neo4j password")
    NEO4J_DATABASE: str = Field(default="neo4j", description="Neo4j database name")
    NEO4J_DEFAULT_GRAPH_QUERY: str = Field(
        default="MATCH (a)-[r]-(b) RETURN a, r, b LIMIT 100",
        description="Default graph query"
    )

    class Config:
        env_file = ".env"   # 从.env读取配置
        case_sensitive = False  # 大小写不敏感
        extra = "ignore"    # 忽略额外的环境变量
    
    # OpenAI compatibility fields
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_BASE: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 设置兼容性字段
        self.OPENAI_API_KEY = self.LLM_API_KEY
        self.OPENAI_API_BASE = self.LLM_BASE_URL
        self.OPENAI_MODEL = self.LLM_MODEL

    def get_cors_origins(self) -> List[str]:
        """获取CORS允许的来源列表"""
        if isinstance(self.CORS_ORIGINS, str):
            return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
        return list(self.CORS_ORIGINS)
    
    # Knowledge base retrieval
    KB_TOP_K: int = 5
    KB_SIMILARITY_THRESHOLD: float = 0.2
    KB_CHUNK_SIZE: int = Field(
        default=512,
        description="Chunk size used when splitting documents for the knowledge base",
    )
    KB_CHUNK_OVERLAP: int = Field(
        default=80,
        description="Chunk overlap used when splitting documents",
    )
    
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "fitness"
    MILVUS_INDEX_TYPE: str = "IVF_FLAT" # 倒排索引（IVF） + 精确计算（FLAT）
    MILVUS_METRIC_TYPE: str = "IP"  # 内积（Inner Product）

    # Embedding configuration
    EMBEDDING_MODEL_NAME: str = Field(default="Qwen/Qwen3-Embedding-0.6B", description="Embedding model name")
    EMBEDDING_API_KEY: Optional[str] = None
    EMBEDDING_BASE_URL: Optional[str] = None
    EMBEDDING_DIMENSION: Optional[int] = Field(default=1024, description="Embedding dimension")

    # Reranker configuration
    ENABLE_RERANK: bool = Field(default=True, description="Enable reranking")
    RERANK_ENABLED: bool = Field(default=True, description="Enable reranking")
    RERANK_PROVIDER: str = Field(default="custom", description="Rerank provider: cohere, jina, voyage, custom")
    RERANK_BASE_URL: Optional[str] = Field(default=None, description="Rerank API base URL")
    RERANK_ENDPOINT: str = Field(default="/rerank", description="Rerank endpoint path")
    RERANK_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3", description="Rerank model name")
    RERANK_API_KEY: Optional[str] = None
    RERANK_MAX_CANDIDATES: int = Field(default=20, description="Max candidates for reranking")
    RERANK_TOP_N: int = Field(default=6, description="Top N results after reranking")
    RERANK_TIMEOUT: int = Field(default=30, description="Rerank API timeout in seconds")
    RERANK_SCORE_FUSION_ALPHA: Optional[float] = Field(default=None, description="Score fusion alpha parameter")

    KB_RERANK_SCORE_THRESHOLD: float = Field(
        default=0.8,
        description="Minimum rerank score required for vector results (e.g., Milvus)",
    )

settings = Settings()
