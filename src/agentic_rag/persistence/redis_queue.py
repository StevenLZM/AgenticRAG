"""Minimal asyncio Redis Streams adapter for duplicate-safe job notifications."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from redis.asyncio import Redis
from redis.exceptions import ResponseError


_PUBLISH_ONCE_SCRIPT = """
local existing = redis.call('HGET', KEYS[1], ARGV[1])
if existing then
  return existing
end
local message_id = redis.call(
  'XADD', KEYS[2], '*',
  'aggregate_id', ARGV[3],
  'enqueued_at', ARGV[2]
)
redis.call('HSET', KEYS[1], ARGV[1], message_id)
return message_id
"""


@dataclass(frozen=True, slots=True)
class StreamMessage:
    """A stream notification identified by its durable aggregate ID."""

    id: str
    aggregate_id: str
    enqueued_at: datetime


class StreamBroker(Protocol):
    async def publish(
        self,
        stream: str,
        aggregate_id: str,
        enqueued_at: datetime,
        dedupe_key: str | None = None,
    ) -> str: ...

    async def consume(
        self, stream: str, group: str, consumer: str, block_ms: int
    ) -> list[StreamMessage]: ...

    async def ack(self, stream: str, group: str, message_id: str) -> None: ...

    async def reclaim(
        self, stream: str, group: str, consumer: str, min_idle_ms: int
    ) -> list[StreamMessage]: ...

    async def dead_letter(
        self, dead_stream: str, message: StreamMessage, reason: str
    ) -> None: ...


class RedisStreamsBroker:
    """Thin redis-py asyncio Streams implementation with explicit ack ownership."""

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def publish(
        self,
        stream: str,
        aggregate_id: str,
        enqueued_at: datetime,
        dedupe_key: str | None = None,
    ) -> str:
        if dedupe_key is None:
            message_id = await self._client.xadd(
                stream,
                {
                    "aggregate_id": aggregate_id,
                    "enqueued_at": _format_timestamp(enqueued_at),
                },
            )
            return _text(message_id)
        message_id = await cast(
            Awaitable[Any],
            self._client.eval(
                _PUBLISH_ONCE_SCRIPT,
                2,
                f"{stream}:dedupe",
                stream,
                dedupe_key,
                _format_timestamp(enqueued_at),
                aggregate_id,
            ),
        )
        return _text(message_id)

    async def consume(
        self, stream: str, group: str, consumer: str, block_ms: int
    ) -> list[StreamMessage]:
        await self._ensure_group(stream, group)
        entries = await self._client.xreadgroup(
            group,
            consumer,
            {stream: ">"},
            count=100,
            block=block_ms,
        )
        return _read_messages(entries)

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        await self._client.xack(stream, group, message_id)

    async def reclaim(
        self, stream: str, group: str, consumer: str, min_idle_ms: int
    ) -> list[StreamMessage]:
        await self._ensure_group(stream, group)
        response = await self._client.xautoclaim(
            stream,
            group,
            consumer,
            min_idle_ms,
            start_id="0-0",
            count=100,
        )
        return [_message(message_id, fields) for message_id, fields in response[1]]

    async def dead_letter(
        self, dead_stream: str, message: StreamMessage, reason: str
    ) -> None:
        await self._client.xadd(
            dead_stream,
            {
                "aggregate_id": message.aggregate_id,
                "enqueued_at": _format_timestamp(message.enqueued_at),
                "source_message_id": message.id,
                "reason": reason,
            },
        )

    async def _ensure_group(self, stream: str, group: str) -> None:
        try:
            await self._client.xgroup_create(stream, group, id="0-0", mkstream=True)
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise


def _read_messages(entries: list[Any]) -> list[StreamMessage]:
    return [
        _message(message_id, fields)
        for _, messages in entries
        for message_id, fields in messages
    ]


def _message(message_id: Any, fields: dict[Any, Any]) -> StreamMessage:
    return StreamMessage(
        id=_text(message_id),
        aggregate_id=_text(_field(fields, "aggregate_id")),
        enqueued_at=datetime.fromisoformat(_text(_field(fields, "enqueued_at"))),
    )


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _field(fields: dict[Any, Any], name: str) -> Any:
    if name in fields:
        return fields[name]
    return fields[name.encode()]
