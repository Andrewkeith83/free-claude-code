"""
Global Rate Limiter for Messaging Platforms.

Centralizes outgoing message requests and ensures compliance with rate limits
using a leaky bucket algorithm (aiolimiter) and a task queue.
"""

import asyncio
import logging
import os
from typing import Awaitable, Callable, Any, Optional, List, Dict
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)


class GlobalRateLimiter:
    """
    A thread-safe global rate limiter for messaging.

    Uses a custom queue with task compaction (deduplication) to ensure
    only the latest version of a message update is processed.
    """

    _instance: Optional["GlobalRateLimiter"] = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            pass
        return super(GlobalRateLimiter, cls).__new__(cls)

    @classmethod
    async def get_instance(cls) -> "GlobalRateLimiter":
        """Get the singleton instance of the limiter."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                # Start the background worker
                asyncio.create_task(cls._instance._worker())
        return cls._instance

    def __init__(self):
        # Prevent double initialization in singleton
        if hasattr(self, "_initialized"):
            return

        rate_limit = int(os.getenv("MESSAGING_RATE_LIMIT", "1"))
        rate_window = float(os.getenv("MESSAGING_RATE_WINDOW", "2.0"))

        self.limiter = AsyncLimiter(rate_limit, rate_window)
        # Custom queue state
        self._queue_list: List[str] = []  # List of dedup_keys in order
        self._queue_map: Dict[
            str, tuple[Callable[[], Awaitable[Any]], List[asyncio.Future]]
        ] = {}
        self._condition = asyncio.Condition()

        self._initialized = True
        self._paused_until = 0

        logger.info(
            f"GlobalRateLimiter initialized ({rate_limit} req / {rate_window}s with Task Compaction)"
        )

    async def _worker(self):
        """Background worker that processes queued messaging tasks."""
        logger.info("GlobalRateLimiter worker started")
        while True:
            try:
                # Get a task from the queue
                async with self._condition:
                    while not self._queue_list:
                        await self._condition.wait()

                    dedup_key = self._queue_list.pop(0)
                    func, futures = self._queue_map.pop(dedup_key)

                # Check for manual pause (FloodWait)
                now = asyncio.get_event_loop().time()
                if self._paused_until > now:
                    wait_time = self._paused_until - now
                    logger.warning(
                        f"Limiter worker paused, waiting {wait_time:.1f}s more..."
                    )
                    await asyncio.sleep(wait_time)

                # Wait for rate limit capacity
                async with self.limiter:
                    try:
                        result = await func()
                        for f in futures:
                            if not f.done():
                                f.set_result(result)
                    except Exception as e:
                        # Report error to all futures and log it
                        for f in futures:
                            if not f.done():
                                f.set_exception(e)

                        error_msg = str(e).lower()
                        if "flood" in error_msg or "wait" in error_msg:
                            seconds = 30
                            try:
                                if hasattr(e, "seconds"):
                                    seconds = e.seconds
                            except:
                                pass

                            logger.error(
                                f"FloodWait detected! Pausing worker for {seconds}s"
                            )
                            self._paused_until = (
                                asyncio.get_event_loop().time() + seconds
                            )
                        else:
                            logger.error(
                                f"Error in limiter worker for key {dedup_key}: {e}"
                            )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in limiter worker: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _enqueue_internal(self, func, future, dedup_key, front=False):
        async def callback(f):
            # This is just a placeholder to use the same internal logic
            pass

        await self._enqueue_internal_multi(func, [future], dedup_key, front)

    async def _enqueue_internal_multi(self, func, futures, dedup_key, front=False):
        async with self._condition:
            if dedup_key in self._queue_map:
                # Compaction: Update existing task with new func, append new futures
                old_func, old_futures = self._queue_map[dedup_key]
                old_futures.extend(futures)
                self._queue_map[dedup_key] = (func, old_futures)
                logger.debug(
                    f"Compacted task for key: {dedup_key} (now {len(old_futures)} futures)"
                )
            else:
                self._queue_map[dedup_key] = (func, futures)
                if front:
                    self._queue_list.insert(0, dedup_key)
                else:
                    self._queue_list.append(dedup_key)
                self._condition.notify_all()

    async def enqueue(
        self, func: Callable[[], Awaitable[Any]], dedup_key: Optional[str] = None
    ) -> Any:
        """
        Enqueue a messaging task and return its future result.
        If dedup_key is provided, subsequent tasks with the same key will replace this one.
        """
        if dedup_key is None:
            # Unique key to avoid deduplication
            dedup_key = f"task_{id(func)}_{asyncio.get_event_loop().time()}"

        future = asyncio.get_event_loop().create_future()
        await self._enqueue_internal(func, future, dedup_key)
        return await future

    def fire_and_forget(
        self, func: Callable[[], Awaitable[Any]], dedup_key: Optional[str] = None
    ):
        """Enqueue a task without waiting for the result."""
        if dedup_key is None:
            dedup_key = f"task_{id(func)}_{asyncio.get_event_loop().time()}"

        future = asyncio.get_event_loop().create_future()

        async def _wrapped():
            try:
                return await self._enqueue_internal(func, future, dedup_key)
            except Exception as e:
                logger.error(f"Error in fire_and_forget for key {dedup_key}: {e}")
                if not future.done():
                    future.set_exception(e)

        asyncio.create_task(_wrapped())
