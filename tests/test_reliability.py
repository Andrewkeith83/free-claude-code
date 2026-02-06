import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from messaging.telegram import TelegramPlatform
from telegram.error import NetworkError, RetryAfter, TelegramError
from messaging.handler import ClaudeMessageHandler, format_status


@pytest.fixture
def telegram_platform():
    with patch("messaging.telegram.TELEGRAM_AVAILABLE", True):
        platform = TelegramPlatform(bot_token="test_token", allowed_user_id="12345")
        return platform


@pytest.mark.asyncio
async def test_telegram_retry_on_network_error(telegram_platform):
    mock_bot = AsyncMock()
    mock_msg = MagicMock()
    mock_msg.message_id = 999

    # Fail twice, then succeed
    mock_bot.send_message.side_effect = [
        NetworkError("Connection failed"),
        NetworkError("Connection failed"),
        mock_msg,
    ]

    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    # We need to patch asyncio.sleep to speed up the test
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        msg_id = await telegram_platform.send_message("chat_1", "hello")

        assert msg_id == "999"
        assert mock_bot.send_message.call_count == 3
        assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_telegram_retry_on_retry_after(telegram_platform):
    mock_bot = AsyncMock()
    mock_msg = MagicMock()
    mock_msg.message_id = 1000

    # Fail with RetryAfter, then succeed
    mock_bot.send_message.side_effect = [RetryAfter(retry_after=5), mock_msg]

    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        msg_id = await telegram_platform.send_message("chat_1", "hello")

        assert msg_id == "1000"
        assert mock_bot.send_message.call_count == 2
        mock_sleep.assert_called_with(5)


@pytest.mark.asyncio
async def test_telegram_no_retry_on_bad_request(telegram_platform):
    mock_bot = AsyncMock()

    # Fail with generic TelegramError (should not retry unless specific conditions met)
    mock_bot.send_message.side_effect = TelegramError("Bad Request: some error")

    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    with pytest.raises(TelegramError):
        await telegram_platform.send_message("chat_1", "hello")

    assert mock_bot.send_message.call_count == 1


def test_handler_build_message_hardening():
    handler = ClaudeMessageHandler(AsyncMock(), AsyncMock(), AsyncMock())

    # Case 1: Empty components
    components = {
        "thinking": [],
        "tools": [],
        "subagents": [],
        "content": [],
        "errors": [],
    }
    msg = handler._build_message(components)
    assert msg == format_status("⏳", "Claude is working...")

    # Case 2: Truncation with code block closing
    long_thinking = "thought " * 200  # ~1400 chars
    components["thinking"] = [long_thinking]
    components["content"] = ["This is a very long message. " * 300]  # ~ 8700 chars

    msg = handler._build_message(components, status="Finishing...")

    assert len(msg) <= 4096
    assert "truncated" in msg
    assert "Finishing..." in msg
    # If thinking contains backticks, they should be balanced
    if "```" in msg:
        assert msg.count("```") % 2 == 0
