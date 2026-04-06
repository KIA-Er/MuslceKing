from __future__ import annotations

import logging
import tempfile
import shutil
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
)

from kb_service.api.deps import get_config
from kb_service.core.config import Config, clone_config
from kb_service.schemas.ingest import ExcelIngestRequest, MySQLIngestRequest
from kb_service.schemas.search import SearchRequest, HybridSearchRequest
from kb_service.services.processor import DataProcessor
from kb_service.services.search import VectorSearcher
from kb_service.services.mysql_ingest import MySQLIngestor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/ingest/excel",
    status_code=status.HTTP_202_ACCEPTED,
    summary="处理本地 Excel 文件并生成向量嵌入到 pgvector 数据库",
)
def ingest_excel(
    payload: ExcelIngestRequest,
    background: BackgroundTasks,
    config: Config = Depends(get_config),
):
    """
    处理本地 Excel 文件：
    - 读取指定路径的 Excel 文件
    - 生成向量嵌入并存储到 pgvector 数据库
    - 支持增量和全量两种处理模式
    """
    excel_path = Path(payload.excel_path)
    if not excel_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Excel file not found: {excel_path}"
        )

    def run_ingest() -> None:
        try:
            local_config = clone_config(config)
            local_config.excel_file_path = str(excel_path)
            processor = DataProcessor(local_config, incremental=payload.incremental)
            processor.process_excel()
        except Exception:  # pragma: no cover - background task logging
            logger.exception("Excel ingest failed for %s", excel_path)

    # 后台任务执行
    background.add_task(run_ingest)
    mode = "增量" if payload.incremental else "全量"
    logger.info(f"Excel ingest queued for {excel_path} (模式: {mode})")
    return {
        "message": "ingest started",
        "path": str(excel_path),
        "mode": "incremental" if payload.incremental else "full",
    }


@router.post(
    "/ingest/excel/upload",
    status_code=status.HTTP_202_ACCEPTED,
    summary="上传 Excel 文件(临时存储)并生成向量嵌入到 pgvector 数据库",
)
async def upload_and_ingest_excel(
    file: UploadFile = File(..., description="Excel file to upload"),
    incremental: bool = Form(False, description="Whether to use incremental mode"),
    background: BackgroundTasks = None,
    config: Config = Depends(get_config),
):
    """
    上传 Excel 文件并处理：
    - 支持主机任意路径的文件上传
    - 文件会保存到临时目录
    - 处理完成后自动清理临时文件
    """
    # 验证文件类型
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="只支持 Excel 文件 (.xlsx, .xls)")

    # 创建临时目录保存上传的文件
    temp_dir = tempfile.mkdtemp()
    temp_file_path = Path(temp_dir) / file.filename

    try:
        # 把上传的文件内容写入临时文件
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info("文件已上传到临时目录: %s", temp_file_path)

        # 定义后台任务
        def run_ingest() -> None:
            try:
                local_config = clone_config(config)
                local_config.excel_file_path = str(temp_file_path)
                processor = DataProcessor(local_config, incremental=incremental)
                processor.process_excel()
                logger.info("Excel 处理完成，文件: %s", temp_file_path)
            except Exception:
                logger.exception("Excel ingest failed for %s", temp_file_path)
            finally:
                # 清理临时文件
                try:
                    shutil.rmtree(temp_dir)
                    logger.info("临时文件已清理: %s", temp_dir)
                except Exception:
                    logger.exception("清理临时文件失败: %s", temp_dir)

        # 添加后台任务
        background.add_task(run_ingest)
        mode = "增量" if incremental else "全量"
        logger.info(f"Excel 上传处理已加入队列: {file.filename} (模式: {mode})")

        return {
            "message": "文件上传成功，处理已开始",
            "filename": file.filename,
            "mode": "incremental" if incremental else "full",
            "size_bytes": temp_file_path.stat().st_size,
        }

    except Exception as e:
        # 如果出错，清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.exception("文件上传失败")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@router.post(
    "/ingest/mysql",
    status_code=status.HTTP_202_ACCEPTED,
    summary="从 MySQL 数据库中提取数据，重写/展开，生成向量嵌入并存储到 pgvector 数据库",
)
def ingest_mysql(
    payload: MySQLIngestRequest,
    background: BackgroundTasks,
    config: Config = Depends(get_config),
):
    """
    从 MySQL 数据库中提取数据，重写/展开，生成向量嵌入并存储到 pgvector 数据库。

    - **payload**: MySQL 数据导入请求
    - **background**: 用于添加后台任务的对象。
    - **config**: 应用配置依赖项。
    """
    # 将模型实例序列化为 Python 字典
    payload_data = payload.model_dump()

    def run_ingest() -> None:
        try:
            local_config = clone_config(config)
            ingestor = MySQLIngestor(local_config)
            ingestor.ingest(MySQLIngestRequest(**payload_data))
        except Exception:  # pragma: no cover - background task logging
            logger.exception("MySQL ingest failed for %s", payload_data.get("table"))

    # 后台任务执行
    background.add_task(run_ingest)
    logger.info("MySQL ingest queued for %s", payload.table)
    return {"message": "ingest started", "table": payload.table}


@router.post(
    "/search",
    status_code=status.HTTP_200_OK,
    summary="在 pgvector 中进行向量检索",
)
def vector_search(payload: SearchRequest, config: Config = Depends(get_config)):
    """
    在 pgvector 数据库中执行向量相似性搜索。

    - **payload**: 包含搜索查询参数的请求体。
    - **config**: 应用配置依赖项。
    """
    searcher = VectorSearcher(config)
    results = searcher.search_similar(
        query=payload.query,
        top_k=payload.top_k,
        threshold=payload.threshold,
        metric=payload.metric,
        company_filter=payload.company_filter,
        source_tables=payload.source_tables,
    )
    return {"query": payload.query, "results": results, "count": len(results)}


@router.post(
    "/search/hybrid",
    status_code=status.HTTP_200_OK,
    summary="在 pgvector 中执行混合搜索（向量检索 + 重排序）",
)
def hybrid_search(payload: HybridSearchRequest, config: Config = Depends(get_config)):
    """
    在 pgvector 数据库中执行混合搜索，结合向量检索和重排序技术。

    - **payload**: 包含混合搜索查询参数的请求体。
    - **config**: 应用配置依赖项。
    """
    searcher = VectorSearcher(config)
    return searcher.hybrid_search(
        query=payload.query,
        vector_top_k=payload.vector_top_k,
        rerank_top_k=payload.rerank_top_k,
        threshold=payload.threshold,
        metric=payload.metric,
        source_tables=payload.source_tables,
    )
