"""
聚合v1版本的所有路由
"""

from fastapi import APIRouter

# from muscleking.app.api.chat import router as chat_router
from muscleking.app.api.health import router as health_router


api_v1_router = APIRouter()

# api_v1_router.include_router(chat_router,) # TODO:调试完lifespan后来看chat接口
api_v1_router.include_router(
    health_router,
)
# api_v1_router.include_router(sessions.router, prefix="/sessions", tags=["Chat Sessions"])
# api_v1_router.include_router(upload.router, prefix="/upload", tags=["File Upload"])
