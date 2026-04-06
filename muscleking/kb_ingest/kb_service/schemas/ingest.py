from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class ExcelIngestRequest(BaseModel):
    """Excel 处理请求"""

    excel_path: str = Field(..., description="需要处理的 Excel 文件的路径。")
    regenerate: bool = Field(
        default=False,
        description="即使之前存在输出，也强制重新生成。",
    )
    incremental: bool = Field(
        default=False,
        description="是否启用增量模式。如果启用，则只处理那些内容发生过变化的行，判断依据是每行内容的哈希值（hash）是否改变。",
    )


class MySQLIngestRequest(BaseModel):
    """MySQL 数据导入请求"""

    connection_url: str = Field(
        ..., description="SQLAlchemy style MySQL connection URL."
    )
    table: str = Field(..., description="Table name to read from.")
    where: Optional[str] = Field(None, description="Optional SQL WHERE clause.")
    limit: Optional[int] = Field(
        None, ge=1, description="Maximum number of rows to ingest."
    )
    prompt_key: Optional[str] = Field(
        None,
        description="Optional prompt template key if not inferred from the table name.",
    )
    extra_metadata: Dict[str, Any] = Field(description="Static metadata to persist.")
    mode: Literal["rewrite", "flatten"] = Field(
        default="rewrite",
        description="Rewrite with LLM or directly flatten the row for embedding.",
    )
    prompt_template: Optional[str] = Field(
        default=None,
        description="Override user prompt template (string.Template syntax).",
    )
    id_column: Optional[str] = Field(
        default=None,
        description="Column to use as unique identifier; defaults to row index if omitted.",
    )
    company_field: Optional[str] = Field(
        default=None, description="Field containing company name (optional)."
    )
    report_year_field: Optional[str] = Field(
        default=None, description="Field containing report year (optional)."
    )
    chunk_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of records to process per batch; defaults to processing all rows at once.",
    )
