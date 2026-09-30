"""Async SQLAlchemy boundary for MySQL persistence."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


MYSQL_SESSION_INIT_COMMAND = "SET time_zone = '+00:00'"


def create_mysql_engine(dsn: str, **options: object) -> AsyncEngine:
    """Create an asyncmy-backed engine with UTC session timestamps."""
    url = make_url(dsn)
    if url.drivername != "mysql+asyncmy":
        raise ValueError("MySQL DSN must use the mysql+asyncmy driver")

    raw_connect_args = options.pop("connect_args", None)
    if raw_connect_args is None:
        connect_args: dict[str, object] = {}
    elif isinstance(raw_connect_args, Mapping):
        connect_args = dict(raw_connect_args)
    else:
        raise TypeError("MySQL connect_args must be a mapping")
    if "init_command" in connect_args:
        raise ValueError("MySQL init_command is reserved for UTC session setup")
    connect_args["init_command"] = MYSQL_SESSION_INIT_COMMAND
    options["connect_args"] = connect_args

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
