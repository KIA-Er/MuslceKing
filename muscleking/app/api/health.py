from fastapi import APIRouter
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, MessagesState, StateGraph

from muscleking.app.core.context import get_global_context
from muscleking.app.models.api_model import HealthResponse


router = APIRouter(tags=["心跳接口"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    ctx = get_global_context()

    # 直接使用 MessagesState，它已经包含了 messages 字段
    def call_model(state: MessagesState):
        response = ctx.llm.invoke(state["messages"])
        return {"messages": response}

    # 使用 MessagesState 作为状态类
    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    checkpointer = ctx.checkpointer
    graph = builder.compile(checkpointer=checkpointer)

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "test_for_checkpointer"
        }
    }

    messages = [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("hi! I'm bob")
    ]

    # 直接传入 messages 字典
    resp = await graph.ainvoke({"messages": messages}, config)
    print(resp)

    return HealthResponse()
