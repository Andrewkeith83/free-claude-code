"""
Telegram Platform Adapter

Implements MessagingPlatform for Telegram using python-telegram-bot.
"""

import asyncio
import logging
import os
from typing import Callable, Awaitable, Optional, Any, Dict

from .base import MessagingPlatform
from .models import IncomingMessage

logger = logging.getLogger(__name__)

# Optional import - python-telegram-bot may not be installed
try:
    from telegram import Update, Bot
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        ContextTypes,
        filters,
    )
    from telegram.error import TelegramError, RetryAfter, NetworkError

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


class TelegramPlatform(MessagingPlatform):
    """
    Telegram messaging platform adapter.

    Uses python-telegram-bot (BoT API) for Telegram access.
    Requires a Bot Token from @BotFather.
    """

    name = "telegram"

    def __init__(
        self,
        bot_token: Optional[str] = None,
        allowed_user_id: Optional[str] = None,
    ):
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot is required. Install with: pip install python-telegram-bot"
            )

        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.allowed_user_id = allowed_user_id or os.getenv("ALLOWED_TELEGRAM_USER_ID")

        if not self.bot_token:
            # We don't raise here to allow instantiation for testing/conditional logic,
            # but start() will fail.
            logger.warning("TELEGRAM_BOT_TOKEN not set")

        self._application: Optional[Application] = None
        self._message_handler: Optional[
            Callable[[IncomingMessage], Awaitable[None]]
        ] = None
        self._connected = False
        self._limiter: Optional[Any] = None  # Will be GlobalRateLimiter

    async def start(self) -> None:
        """Initialize and connect to Telegram."""
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        # Build Application
        builder = Application.builder().token(self.bot_token)
        self._application = builder.build()

        # Register Internal Handlers
        # We catch ALL text messages and commands to forward them
        self._application.add_handler(
            MessageHandler(filters.TEXT & (~filters.COMMAND), self._on_telegram_message)
        )
        self._application.add_handler(CommandHandler("start", self._on_start_command))
        # Catch-all for other commands if needed, or let them fall through
        self._application.add_handler(
            MessageHandler(filters.COMMAND, self._on_telegram_message)
        )

        # Initialize internal components
        await self._application.initialize()
        await self._application.start()

        # Start polling (non-blocking way for integration)
        # allowed_updates=None (all)
        await self._application.updater.start_polling(drop_pending_updates=False)

        self._connected = True

        # Initialize rate limiter
        from .limiter import GlobalRateLimiter

        self._limiter = await GlobalRateLimiter.get_instance()

        # Send startup notification
        try:
            target = self.allowed_user_id
            if target:
                await self.send_message(
                    target, "🚀 **Claude Code Proxy is online!** (Bot API)"
                )
        except Exception as e:
            logger.warning(f"Could not send startup message: {e}")

        logger.info("Telegram platform started (Bot API)")

    async def stop(self) -> None:
        """Stop the bot."""
        if self._application:
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()

        self._connected = False
        logger.info("Telegram platform stopped")

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
        parse_mode: Optional[str] = "Markdown",
    ) -> str:
        """Send a message to a chat."""
        if not self._application:
            raise RuntimeError("Telegram application not initialized")

        try:
            msg = await self._application.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=int(reply_to) if reply_to else None,
                parse_mode=parse_mode,
            )
            return str(msg.message_id)
        except RetryAfter as e:
            logger.warning(f"Rate limited by Telegram, retry in {e.retry_after}s")
            # In a real system, we might sleep, but for now we propagate or let the caller handle
            raise
        except TelegramError as e:
            if "Can't parse entities" in str(e):
                logger.warning(
                    f"Markdown failed for message, falling back to plain text: {e}"
                )
                # Retry as plain text (no parse_mode)
                msg = await self._application.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_to_message_id=int(reply_to) if reply_to else None,
                    parse_mode=None,
                )
                return str(msg.message_id)
            logger.error(f"Telegram API Error: {e}")
            raise

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: Optional[str] = "Markdown",
    ) -> None:
        """Edit an existing message."""
        if not self._application:
            raise RuntimeError("Telegram application not initialized")

        try:
            await self._application.bot.edit_message_text(
                chat_id=chat_id,
                message_id=int(message_id),
                text=text,
                parse_mode=parse_mode,
            )
        except RetryAfter as e:
            logger.warning(f"Rate limited on edit, retry in {e.retry_after}s")
            raise
        except TelegramError as e:
            if "Message is not modified" in str(e):
                pass
            elif "Can't parse entities" in str(e):
                logger.warning(
                    f"Markdown failed for edit, falling back to plain text: {e}"
                )
                # Retry as plain text
                await self._application.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=int(message_id),
                    text=text,
                    parse_mode=None,
                )
            else:
                logger.error(f"Telegram Edit Error: {e}")
                raise

    async def queue_send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
        parse_mode: Optional[str] = "Markdown",
        fire_and_forget: bool = True,
    ) -> Optional[str]:
        """Enqueue a message to be sent (using limiter)."""
        # Note: Bot API handles limits better, but we still use our limiter for nice queuing
        if not self._limiter:
            return await self.send_message(chat_id, text, reply_to, parse_mode)

        async def _send():
            return await self.send_message(chat_id, text, reply_to, parse_mode)

        if fire_and_forget:
            self._limiter.fire_and_forget(_send)
            return None
        else:
            return await self._limiter.enqueue(_send)

    async def queue_edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: Optional[str] = "Markdown",
        fire_and_forget: bool = True,
    ) -> None:
        """Enqueue a message edit."""
        if not self._limiter:
            return await self.edit_message(chat_id, message_id, text, parse_mode)

        async def _edit():
            return await self.edit_message(chat_id, message_id, text, parse_mode)

        dedup_key = f"edit:{chat_id}:{message_id}"
        if fire_and_forget:
            self._limiter.fire_and_forget(_edit, dedup_key=dedup_key)
        else:
            await self._limiter.enqueue(_edit, dedup_key=dedup_key)

    def fire_and_forget(self, task: Awaitable[Any]) -> None:
        """Execute a coroutine without awaiting it."""
        asyncio.create_task(task)

    def on_message(
        self,
        handler: Callable[[IncomingMessage], Awaitable[None]],
    ) -> None:
        """Register a message handler callback."""
        self._message_handler = handler

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    async def _on_start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command."""
        await update.message.reply_text("👋 Hello! I am the Claude Code Proxy Bot.")
        # We can also treat this as a message if we want it to trigger something
        await self._on_telegram_message(update, context)

    async def _on_telegram_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming updates."""
        if not update.message or not update.message.text:
            return

        user_id = str(update.effective_user.id)
        chat_id = str(update.effective_chat.id)

        # Security check
        if self.allowed_user_id:
            if user_id != str(self.allowed_user_id).strip():
                logger.warning(f"Unauthorized access attempt from {user_id}")
                return

        if not self._message_handler:
            return

        incoming = IncomingMessage(
            text=update.message.text,
            chat_id=chat_id,
            user_id=user_id,
            message_id=str(update.message.message_id),
            platform="telegram",
            reply_to_message_id=str(update.message.reply_to_message.message_id)
            if update.message.reply_to_message
            else None,
            raw_event=update,
        )

        try:
            await self._message_handler(incoming)
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            try:
                await self.send_message(
                    chat_id,
                    f"❌ **Error:** {str(e)[:200]}",
                    reply_to=incoming.message_id,
                )
            except Exception:
                pass
