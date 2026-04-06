from fastapi import APIRouter

from muscleking.app.models.api_model import HealthResponse


router = APIRouter(tags=["心跳接口"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse()
