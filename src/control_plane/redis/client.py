"""
RedisClient — thin wrapper around redis-py that reads connection settings
from environment variables and provides a shared connection pool.
"""
from __future__ import annotations

import os
import redis
import redis.asyncio as aioredis


_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class RedisClient:
    """Synchronous Redis client backed by a connection pool."""

    def __init__(self, url: str = _REDIS_URL) -> None:
        self._pool = redis.ConnectionPool.from_url(url, decode_responses=True)
        self._client = redis.Redis(connection_pool=self._pool)

    @property
    def r(self) -> redis.Redis:
        return self._client

    def ping(self) -> bool:
        return self._client.ping()

    def close(self) -> None:
        self._pool.disconnect()

    # Context-manager support
    def __enter__(self) -> "RedisClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()


class AsyncRedisClient:
    """Async Redis client for use in asyncio worker loops."""

    def __init__(self, url: str = _REDIS_URL) -> None:
        self._url = url
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._client = await aioredis.from_url(self._url, decode_responses=True)

    @property
    def r(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("AsyncRedisClient not connected — call connect() first")
        return self._client

    async def ping(self) -> bool:
        return await self.r.ping()

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncRedisClient":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
