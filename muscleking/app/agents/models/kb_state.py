"""
知识库状态模型。
"""

from pydantic import BaseModel
from typing import Optional, Literal, List, Dict, Any, TypedDict, Annotated
from operator import add
from pydantic import Field


class KBGuardrailsDecision(BaseModel):  # 安全护栏决策
    decision: Literal["proceed", "end"]
    summary: Optional[str] = None  # 安全护栏决策摘要
    rationale: Optional[str] = None  # 安全护栏决策理由


class KBRouteDecision(BaseModel):  # 路由决策
    route: Literal["local", "external", "hybrid"]
    rationale: str  # 路由决策理由
    tools: List[Literal["milvus", "postgres"]] = Field(
        description="本地知识源检索工具列表，支持 milvus/postgres",
    )  # 路由决策使用的本地知识源检索工具列表


class KBInputState(TypedDict):
    question: str  # 用户问题
    history: List[Dict[str, str]]  # 对话历史记录


class KBWorkflowState(TypedDict):
    question: str  # 用户问题
    history: List[Dict[str, str]]  # 对话历史记录
    guardrails_decision: str  # 安全护栏决策
    summary: str  # 推理结果摘要
    route: str  # 路由决策，选择本地、外部或混合知识源
    kb_tools: List[str]  # 知识库工具列表
    milvus_results: List[Dict[str, Any]]  # milvus 检索结果
    postgres_results: List[Dict[str, Any]]  # postgres 检索结果
    local_results: List[Dict[str, Any]]  # 本地知识源检索结果
    external_results: List[Dict[str, Any]]  # 外部知识源检索结果
    answer: str  # 最终回答
    steps: Annotated[List[str], add]  # 当前执行的所有步骤
    sources: Annotated[List[str], add]  # 引用来源


class KBOutputState(TypedDict):
    answer: str  # 最终回答
    steps: List[str]  # 执行的所有步骤
    sources: List[str]  # 引用来源
