"""Async SQLAlchemy boundary for MySQL persistence."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_mysql_engine(dsn: str, **options: object) -> AsyncEngine:
    """Create an asyncmy-backed engine without opening a connection."""
    url = make_url(dsn)
    if url.drivername != "mysql+asyncmy":
        raise ValueError("MySQL DSN must use the mysql+asyncmy driver")
    return create_async_engine(url, **options)


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create sessions whose objects remain readable after transaction commit."""
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def transaction(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Yield one caller-owned transaction shared by cooperating repositories."""
    async with factory.begin() as session:
        yield session
