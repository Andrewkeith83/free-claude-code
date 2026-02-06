import pytest
from unittest.mock import MagicMock
from messaging.handler import ClaudeMessageHandler, escape_md_v2


@pytest.fixture
def handler():
    platform = MagicMock()
    cli = MagicMock()
    store = MagicMock()
    return ClaudeMessageHandler(platform, cli, store)


def test_truncation_closes_code_blocks(handler):
    """Verify that truncation correctly closes open code blocks."""
    components = {
        "thinking": [
            "Starting some long thinking process that will definitely cause truncation later on..."
        ],
        "tools": [],
        "subagents": [],
        "content": [
            "```python\ndef very_long_function():\n    # " + "A" * 4000
        ],  # Long content
        "errors": [],
    }

    msg = handler._build_message(components, "✅ *Complete*")

    assert escape_md_v2("... (truncated)") in msg
    # The limit is 3900. Our content + thinking is > 4000.
    # The backtick count must be even to be a valid block.
    assert msg.count("```") % 2 == 0
    assert msg.endswith("```") or "✅ *Complete*" in msg.split("```")[-1]


def test_truncation_preserves_status(handler):
    """Verify that status is still appended after truncation."""
    components = {
        "thinking": ["Thinking..."],
        "tools": [],
        "subagents": [],
        "content": ["A" * 5000],
        "errors": [],
    }
    status = "READY_STATUS"
    msg = handler._build_message(components, status)

    assert status in msg
    assert escape_md_v2("... (truncated)") in msg


def test_empty_components_with_status(handler):
    """Verify message building with just a status."""
    components = {
        "thinking": [],
        "tools": [],
        "subagents": [],
        "content": [],
        "errors": [],
    }
    status = "Simple Status"
    msg = handler._build_message(components, status)
    assert msg == "\n\nSimple Status"
