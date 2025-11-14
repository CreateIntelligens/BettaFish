"""
通用數據庫工具（異步）

此模塊提供基於 SQLAlchemy 2.x 異步引擎的數據庫訪問封裝，支持 MySQL 與 PostgreSQL。
數據模型定義位置：
- 無（本模塊僅提供連接與查詢工具，不定義數據模型）
"""

from __future__ import annotations
from urllib.parse import quote_plus
import asyncio
import os
from typing import Any, Dict, Iterable, List, Optional, Union

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy import text
from InsightEngine.utils.config import settings

__all__ = [
    "get_async_engine",
    "fetch_all",
]


_engine: Optional[AsyncEngine] = None


def _build_database_url() -> str:
    dialect: str = (settings.DB_DIALECT or "mysql").lower()
    host: str = settings.DB_HOST or ""
    port: str = str(settings.DB_PORT or "")
    user: str = settings.DB_USER or ""
    password: str = settings.DB_PASSWORD or ""
    db_name: str = settings.DB_NAME or ""

    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")  # 直接使用外部提供的完整URL

    password = quote_plus(password)

    if dialect in ("postgresql", "postgres"):
        # PostgreSQL 使用 asyncpg 驅動
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"

    # 默認 MySQL 使用 aiomysql 驅動
    return f"mysql+aiomysql://{user}:{password}@{host}:{port}/{db_name}"


def get_async_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        database_url: str = _build_database_url()
        _engine = create_async_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
    return _engine


async def fetch_all(query: str, params: Optional[Union[Iterable[Any], Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    執行只讀查詢並返回字典列表。
    """
    engine: AsyncEngine = get_async_engine()
    async with engine.connect() as conn:
        result = await conn.execute(text(query), params or {})
        rows = result.mappings().all()
        # 將 RowMapping 轉換爲普通字典
        return [dict(row) for row in rows]


