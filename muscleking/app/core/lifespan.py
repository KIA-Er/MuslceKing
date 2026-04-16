from contextlib import asynccontextmanager
from pathlib import Path
from loguru import logger
from fastapi import FastAPI
import os

from muscleking.app.config.settings import get_settings
from muscleking.app.core.context import AppContext, get_global_context, init_global_context
from muscleking.app.utils.banner_config import start_banner
from muscleking.app.resource.factories.middleware_factory import MiddlewareFactory
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

ROOT_PATH = Path(__file__).parents[2]
MUSCLEKING_BANNER = (ROOT_PATH / "app/utils/banner.txt").read_text()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动事件"""

    # 使用全局单例获取配置
    settings = get_settings()

    start_banner(MUSCLEKING_BANNER, "系统启动中")

    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("API docs available at http://{}:{}/docs", settings.HOST, settings.PORT)

    try:

        async with AsyncPostgresSaver.from_conn_string(settings.POSTGRES_CHECKPOINT_URI) as checkpointer:
            logger.debug("🔧 初始化 PostgreSQL checkpointer 表结构...")
            await checkpointer.setup()

            init_global_context(settings)
            ctx = get_global_context()
            ctx.checkpointer = checkpointer

            yield

    except Exception as e:
        logger.error(f"❌ {settings.APP_NAME} 启动失败: {e}")
        raise e

    finally:
        """应用关闭事件"""
        logger.info("✅ {} 已安全关闭", settings.APP_NAME)
