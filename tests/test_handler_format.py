import pytest
from unittest.mock import MagicMock
from messaging.handler import ClaudeMessageHandler, escape_md_v2


@pytest.fixture
def handler():
    platform = MagicMock()
    cli = MagicMock()
    store = MagicMock()
    return ClaudeMessageHandler(platform, cli, store)


def test_build_message_structure(handler):
    """Verify the order of message components."""
    components = {
        "thinking": ["Thinking process..."],
        "tools": ["list_files", "read_file"],
        "subagents": ["Searching codebase...", "Analyzing dependencies..."],
        "content": ["Here is the file content."],
        "errors": ["Some error happened"],
    }
    status = "✅ *Complete*"

    msg = handler._build_message(components, status)

    print(f"Generated Message:\n{msg}")

    # Check existence
    assert "Thinking process..." in msg
    assert "list_files" in msg
    assert "read_file" in msg
    assert "Searching codebase..." in msg
    assert escape_md_v2("Here is the file content.") in msg
    assert "Some error happened" in msg
    assert "✅ *Complete*" in msg

    # Check headers
    assert "💭 *Thinking:*" in msg
    assert "🛠 *Tools:*" in msg
    assert "🤖 *Subagent:*" in msg
    assert "⚠️ *Error:*" in msg

    # Check Order: Thinking -> Tools -> Subagents -> Content -> Errors -> Status
    p_thinking = msg.find("Thinking process...")
    p_tools = msg.find("🛠 *Tools:*")
    p_subagents = msg.find("🤖 *Subagent:*")
    p_content = msg.find(escape_md_v2("Here is the file content."))
    p_errors = msg.find("⚠️ *Error:*")
    p_status = msg.find("✅ *Complete*")

    assert p_thinking < p_tools, "Thinking should come before Tools"
    assert p_tools < p_subagents, "Tools should come before Subagents"
    assert p_subagents < p_content, "Subagents should come before Content"
    assert p_content < p_errors, "Content should come before Errors"
    assert p_errors < p_status, "Errors should come before Status"


def test_build_message_simple(handler):
    """Verify simple message with just content."""
    components = {
        "thinking": [],
        "tools": [],
        "subagents": [],
        "content": ["Simple message."],
        "errors": [],
    }
    msg = handler._build_message(components, "Ready")

    assert escape_md_v2("Simple message.") in msg
    assert "Ready" in msg
    assert "💭" not in msg
    assert "🛠" not in msg


def test_subagents_formatting(handler):
    """Verify subagents formatting."""
    components = {
        "thinking": [],
        "tools": [],
        "subagents": ["Task 1", "Task 2"],
        "content": [],
        "errors": [],
    }
    msg = handler._build_message(components)

    assert "🤖 *Subagent:* `Task 1`" in msg
    assert "🤖 *Subagent:* `Task 2`" in msg
