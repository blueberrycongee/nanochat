#!/usr/bin/env python3
"""
Unit tests for OpenAI-compatible API additions.
Run with: python -m pytest tests/test_openai_api.py -v
"""

import pytest
import asyncio
import time
from fastapi import HTTPException

# Import the modules we added
from scripts.chat_web import (
    RateLimiter,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIMessage,
    OpenAIChoice,
    OpenAIChoiceMessage,
    OpenAIUsage,
    OpenAIStreamChunk,
    OpenAIStreamChoice,
    OpenAIStreamDelta,
    OpenAIModel,
    OpenAIModelList,
    validate_openai_request,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW,
)


class TestRateLimiter:
    """Test the RateLimiter class."""
    
    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self):
        """Should allow requests under the limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        for i in range(5):
            assert await limiter.is_allowed("client1") == True
    
    @pytest.mark.asyncio
    async def test_blocks_requests_over_limit(self):
        """Should block requests over the limit."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        # Use up the limit
        for i in range(3):
            await limiter.is_allowed("client1")
        
        # Next request should be blocked
        assert await limiter.is_allowed("client1") == False
    
    @pytest.mark.asyncio
    async def test_different_clients_separate_limits(self):
        """Different clients should have separate limits."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        # Client 1 uses up limit
        await limiter.is_allowed("client1")
        await limiter.is_allowed("client1")
        assert await limiter.is_allowed("client1") == False
        
        # Client 2 should still be allowed
        assert await limiter.is_allowed("client2") == True
    
    def test_get_remaining(self):
        """Should correctly report remaining requests."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.get_remaining("client1") == 5


class TestOpenAIModels:
    """Test OpenAI-compatible Pydantic models."""
    
    def test_openai_message_creation(self):
        """Should create OpenAI message."""
        msg = OpenAIMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_openai_request_defaults(self):
        """Should have correct defaults."""
        request = OpenAIChatRequest(
            messages=[OpenAIMessage(role="user", content="Hi")]
        )
        assert request.model == "nanochat"
        assert request.stream == False
        assert request.temperature is None
        assert request.n == 1
    
    def test_openai_response_creation(self):
        """Should create valid response."""
        response = OpenAIChatResponse(
            id="chatcmpl-123",
            created=int(time.time()),
            model="nanochat-sft",
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIChoiceMessage(content="Hello!"),
                    finish_reason="stop"
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15
            )
        )
        assert response.object == "chat.completion"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello!"
    
    def test_stream_chunk_creation(self):
        """Should create valid stream chunk."""
        chunk = OpenAIStreamChunk(
            id="chatcmpl-123",
            created=int(time.time()),
            model="nanochat-sft",
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIStreamDelta(content="Hi"),
                    finish_reason=None
                )
            ]
        )
        assert chunk.object == "chat.completion.chunk"
    
    def test_model_list_creation(self):
        """Should create valid model list."""
        model_list = OpenAIModelList(
            data=[
                OpenAIModel(
                    id="nanochat-sft",
                    created=int(time.time()),
                    owned_by="nanochat"
                )
            ]
        )
        assert model_list.object == "list"
        assert len(model_list.data) == 1


class TestValidation:
    """Test request validation."""
    
    def test_empty_messages_rejected(self):
        """Should reject empty messages."""
        request = OpenAIChatRequest(messages=[])
        with pytest.raises(HTTPException) as exc_info:
            validate_openai_request(request)
        assert exc_info.value.status_code == 400
    
    def test_valid_request_passes(self):
        """Should pass valid request."""
        request = OpenAIChatRequest(
            messages=[
                OpenAIMessage(role="user", content="Hello")
            ]
        )
        # Should not raise
        validate_openai_request(request)
    
    def test_system_role_allowed(self):
        """Should allow system role in OpenAI format."""
        request = OpenAIChatRequest(
            messages=[
                OpenAIMessage(role="system", content="You are helpful."),
                OpenAIMessage(role="user", content="Hello")
            ]
        )
        # Should not raise
        validate_openai_request(request)
    
    def test_invalid_role_rejected(self):
        """Should reject invalid role."""
        request = OpenAIChatRequest(
            messages=[
                OpenAIMessage(role="invalid_role", content="Hello")
            ]
        )
        with pytest.raises(HTTPException) as exc_info:
            validate_openai_request(request)
        assert exc_info.value.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
