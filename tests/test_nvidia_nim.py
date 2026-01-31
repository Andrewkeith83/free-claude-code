import pytest
import json
import httpx
from unittest.mock import MagicMock, AsyncMock, patch
from providers.nvidia_nim import (
    NvidiaNimProvider,
    RateLimitError,
    APIError,
    OverloadedError,
)
from providers.base import ProviderConfig
from providers.utils import ContentType


# Mock data classes
class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "test-model"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = ["STOP"]
        self.tools = []
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockThinking:
    def __init__(self, enabled=True, budget_tokens=1000):
        self.enabled = enabled
        self.budget_tokens = budget_tokens


@pytest.fixture
def mock_httpx_ssl():
    """Mock ssl context for older httpx versions if needed, or just bypass."""
    # This is often needed if the code under test creates an SSL context
    pass


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.nvidia_nim.GlobalRateLimiter") as mock:
        instance = mock.get_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)
        yield instance


@pytest.mark.asyncio
async def test_init(provider_config):
    """Test provider initialization."""
    provider = NvidiaNimProvider(provider_config)
    assert provider._api_key == "test_key"
    assert provider._base_url == "https://test.api.nvidia.com/v1"


@pytest.mark.asyncio
async def test_build_request_body(nim_provider):
    """Test request body construction."""
    req = MockRequest()
    body = nim_provider._build_request_body(req, stream=True)

    assert body["model"] == "test-model"
    assert body["stream"] is True
    assert body["temperature"] == 0.5
    assert len(body["messages"]) == 2  # System + User
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "System prompt"
    assert "thinking" in body
    assert body["thinking"]["type"] == "enabled"


@pytest.mark.asyncio
async def test_build_request_body_with_tools(nim_provider):
    """Test request body with tools."""
    tool = MockTool(
        name="test_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {"arg": {"type": "string"}}},
    )
    req = MockRequest(tools=[tool])
    body = nim_provider._build_request_body(req)

    assert "tools" in body
    assert len(body["tools"]) == 1
    assert body["tools"][0]["function"]["name"] == "test_tool"


@pytest.mark.asyncio
async def test_build_request_body_deepseek(nim_provider):
    """Test request body with DeepSeek model."""
    req = MockRequest(model="deepseek-ai/deepseek-r1")
    body = nim_provider._build_request_body(req)

    assert "chat_template_kwargs" in body
    assert body["chat_template_kwargs"] == {"thinking": True}


@pytest.mark.asyncio
async def test_build_request_body_non_deepseek(nim_provider):
    """Test request body with non-DeepSeek model."""
    req = MockRequest(model="meta/llama-3.3-70b-instruct")
    body = nim_provider._build_request_body(req)

    assert "chat_template_kwargs" not in body


@pytest.mark.asyncio
async def test_stream_response_text(nim_provider):
    """Test streaming text response."""
    # Mock stream response
    req = MockRequest()

    mock_chunks = [
        'data: {"id":"1","choices":[{"delta":{"content":"Hello"}}]}\n\n',
        'data: {"id":"1","choices":[{"delta":{"content":" World"}}]}\n\n',
        "data: [DONE]\n\n",
    ]

    # Create a mock response that yields chunks
    mock_response = AsyncMock()
    mock_response.status_code = 200

    async def mock_aiter():
        for chunk in mock_chunks:
            yield chunk

    mock_response.aiter_text = mock_aiter
    mock_response.aread = AsyncMock(return_value=b"")

    # Mock the stream context manager
    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__.return_value = mock_response

    with patch.object(nim_provider._client, "stream", return_value=mock_stream_ctx):
        events = []
        async for event in nim_provider.stream_response(req):
            events.append(event)

        # Verify events
        assert len(events) > 0
        assert "event: message_start" in events[0]

        # Check text content
        text_content = ""
        for e in events:
            if "event: content_block_delta" in e and '"text_delta"' in e:
                # Extract json from data: ...
                for line in e.splitlines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "delta" in data and "text" in data["delta"]:
                            text_content += data["delta"]["text"]

        assert "Hello World" in text_content


@pytest.mark.asyncio
async def test_stream_response_thinking_interleaved(nim_provider):
    """Test streaming with interleaved thinking tags."""
    req = MockRequest()

    mock_chunks = [
        'data: {"choices":[{"delta":{"content":"<think>Thinking process..."}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"...</think>Answer"}}]}\n\n',
        "data: [DONE]\n\n",
    ]

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.aiter_text = lambda: (
        c for c in mock_chunks
    )  # sync iterator wrapper check? No, must be async generator

    async def mock_aiter():
        for chunk in mock_chunks:
            yield chunk

    mock_response.aiter_text = mock_aiter

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__.return_value = mock_response

    with patch.object(nim_provider._client, "stream", return_value=mock_stream_ctx):
        events = []
        async for event in nim_provider.stream_response(req):
            events.append(event)

        # Should have thinking events
        think_deltas = [
            e
            for e in events
            if "event: content_block_delta" in e and '"thinking_delta"' in e
        ]
        assert len(think_deltas) > 0

        # Should have text events
        text_deltas = [
            e
            for e in events
            if "event: content_block_delta" in e and '"text_delta"' in e
        ]
        assert len(text_deltas) > 0


@pytest.mark.asyncio
async def test_stream_response_error_429(nim_provider):
    """Test 429 Rate Limit error."""
    req = MockRequest()

    mock_response = AsyncMock()
    mock_response.status_code = 429
    mock_response.aread = AsyncMock(
        return_value=b'{"error": {"message": "Too many requests"}}'
    )

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__.return_value = mock_response

    with patch.object(nim_provider._client, "stream", return_value=mock_stream_ctx):
        # The provider yields an error event rather than raising
        events = []
        async for event in nim_provider.stream_response(req):
            events.append(event)

        # Check for error message in text delta
        found_error = False
        for e in events:
            if "event: content_block_delta" in e and '"text_delta"' in e:
                if "Too many requests" in e:
                    found_error = True
                    break
        assert found_error


@pytest.mark.asyncio
async def test_stream_response_timeout(nim_provider):
    """Test timeout handling."""
    req = MockRequest()

    # Mock stream to raise TimeoutException
    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__.side_effect = httpx.ConnectTimeout(
        "Connection timed out"
    )

    with patch.object(nim_provider._client, "stream", return_value=mock_stream_ctx):
        events = []
        async for event in nim_provider.stream_response(req):
            events.append(event)

        # Check for error message in text delta
        found_error = False
        for e in events:
            if "event: content_block_delta" in e and '"text_delta"' in e:
                if "Timeout" in e:
                    found_error = True
                    break
        assert found_error


@pytest.mark.asyncio
async def test_complete_success(nim_provider):
    """Test successful completion."""
    req = MockRequest()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "test_id",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello world"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    with patch.object(
        nim_provider._client, "post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.return_value = mock_response

        result = await nim_provider.complete(req)
        assert result["id"] == "test_id"
        assert result["choices"][0]["message"]["content"] == "Hello world"


@pytest.mark.asyncio
async def test_complete_error_500(nim_provider):
    """Test 500 error on completion."""
    req = MockRequest()

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = '{"error": {"message": "Internal Server Error"}}'

    with patch.object(
        nim_provider._client, "post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.return_value = mock_response

        with pytest.raises(APIError) as exc:
            await nim_provider.complete(req)
        assert "Internal Server Error" in str(exc.value)


@pytest.mark.asyncio
async def test_convert_response(nim_provider):
    """Test response conversion."""
    req = MockRequest()
    openai_resp = {
        "id": "resp_1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Response text",
                    "reasoning_content": "Found logic",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20},
    }

    result = nim_provider.convert_response(openai_resp, req)

    assert result["id"] == "resp_1"
    assert result["type"] == "message"
    assert len(result["content"]) == 2
    assert result["content"][0]["type"] == "thinking"
    assert result["content"][0]["thinking"] == "Found logic"
    assert result["content"][1]["type"] == "text"
    assert result["content"][1]["text"] == "Response text"


@pytest.mark.asyncio
async def test_tool_call_stream(nim_provider):
    """Test streaming tool calls."""
    req = MockRequest()

    mock_chunks = [
        'data: {"choices":[{"delta":{"content":null,"tool_calls":[{"index":0,"id":"call_1","function":{"name":"search"}}]}}]}\n\n',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\": "}}]}}]}\n\n',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"test\\"}"}}]}}]}\n\n',
        "data: [DONE]\n\n",
    ]

    async def mock_aiter():
        for chunk in mock_chunks:
            yield chunk

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.aiter_text = mock_aiter

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__.return_value = mock_response

    with patch.object(nim_provider._client, "stream", return_value=mock_stream_ctx):
        events = []
        async for event in nim_provider.stream_response(req):
            events.append(event)

        # Should have content_block_start for tool_use
        # Looking for event: content_block_start ... type: tool_use
        starts = [
            e for e in events if "event: content_block_start" in e and '"tool_use"' in e
        ]
        assert len(starts) == 1
        assert "search" in starts[0]
