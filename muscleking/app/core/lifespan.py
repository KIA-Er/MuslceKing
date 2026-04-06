from contextlib import asynccontextmanager
from pathlib import Path
from venv import logger

from fastapi import FastAPI

from muscleking.app.config import settings
from muscleking.app.utils.banner_config import start_banner

ROOT_PATH = Path(__file__).parents[2]
MUSCLEKING_BANNER = (ROOT_PATH / "app/utils/banner.txt").read_text()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动事件"""

    start_banner(MUSCLEKING_BANNER, "系统启动中")

    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("API docs available at http://{}:{}/docs", settings.HOST, settings.PORT)

    try:
        # 后续实现启动时自动创建数据库表

        yield
    except Exception as e:
        logger.error("❌MuscleKing启动失败")
        raise e
    finally:
        """应用关闭事件"""
        logger.info("Shutting down {}", settings.APP_NAME)
        # 后续实现关闭时的清理工作，如数据库连接池关闭等
        pass
