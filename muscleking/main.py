"""
MuscleKing FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from muscleking.app.config.settings import settings
from muscleking.app.api import api_v1_router
from muscleking.app.core import lifespan

# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="智能健身助手，为用户提供健身相关的咨询与指导，以及制定个性化的健身计划。",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,  # TODO:重构todo
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由到/api/v1下
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)
# app.include_router(knowledge_router.router, prefix=settings.API_V1_PREFIX)
# app.include_router(lightrag_router.router, prefix=settings.API_V1_PREFIX)


@app.get("/api")
async def root() -> dict:
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "muscleking.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
