from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from muscleking.app.config.settings import settings


class HealthResponse(BaseModel):
    name: str = Field(default="MuscleKing", description="服务名称")
    status: str = Field(default="healthy", description="服务状态")
    version: str = Field(default=settings.APP_VERSION)
    datetime: str = Field(
        default_factory=lambda: datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
            "%Y-%m-%d, %H:%M:%S %Z"
        ),
        description="当前时间",
    )
