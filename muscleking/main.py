"""
MuscleKing FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from muscleking.config.settings import settings
from loguru import logger
from muscleking.app.api.v1 import api_v1_router

# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="智能健身助手，为用户提供健身相关的咨询与指导，以及制定个性化的健身计划。",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
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

@app.get("/health")
async def health_check() -> dict:
    return {"status": "healthy", "version": settings.APP_VERSION}

@app.on_event("startup")
async def startup_event() -> None:
    """应用启动事件"""
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("Debug mode: {}", settings.DEBUG)
    logger.info("API docs available at http://{}:{}/docs", settings.HOST, settings.PORT)

    # 后续实现启动时自动创建数据库表
    pass

@app.on_event("shutdown")
async def shutdown_event() -> None:
    """应用关闭事件"""
    logger.info("Shutting down {}", settings.APP_NAME)
    # 后续实现关闭时的清理工作，如数据库连接池关闭等
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "muscleking.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
