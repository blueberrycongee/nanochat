#!/usr/bin/env python3
"""
Demo web chat server - Mock version for frontend demonstration.

This is a lightweight demo server that mimics the nanochat API without requiring
a trained model. Useful for frontend UI demonstration, API integration testing,
and interview presentations. The server provides mock responses to showcase the
UI features and API endpoints, but does not perform actual AI inference.

Launch:
    python demo_server.py
    python demo_server.py --port 8080

Endpoints:
    GET  /                    - Chat UI
    POST /chat/completions    - Chat API (legacy, streaming only)
    POST /v1/chat/completions - OpenAI-compatible Chat API (streaming & non-streaming)
    GET  /v1/models           - List available models (OpenAI-compatible)
    GET  /health              - Health check
    GET  /stats               - Server statistics
"""

import argparse
import asyncio
import json
import logging
import os
import random
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='NanoChat Demo Server')
parser.add_argument('-p', '--port', type=int, default=8080, help='Port to run the server on (default: 8080)')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to (default: 0.0.0.0)')
args = parser.parse_args()

app = FastAPI(title="NanoChat Demo", version="1.0.0")

# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.8
    max_tokens: Optional[int] = 512
    top_k: Optional[int] = 50
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

# -----------------------------------------------------------------------------
# Mock Response Generator
# -----------------------------------------------------------------------------

MOCK_RESPONSES = [
    "你好！我是 Nanochat 的演示版本。这是一个 mock 响应，用于展示前端功能。",
    "这是一个演示服务器，不包含真实的 AI 模型。但你可以看到完整的 UI 交互和流式输出效果。",
    "Temperature 参数控制生成的随机性。在真实模型中，较低的值会产生更确定的输出。",
    "Top-K 参数限制了每步采样时考虑的候选词数量。这影响输出的多样性。",
    "Max Tokens 参数限制了生成的最大长度。在真实模型中，这会在达到限制时停止生成。",
    "点击右上角的齿轮图标可以调整这些参数。这些设置会在下次请求时生效。",
    "这个 UI 支持多轮对话。你可以继续提问，系统会记住之前的对话历史。",
    "流式输出让响应逐字显示，提供更好的用户体验。这是通过 Server-Sent Events 实现的。",
]

def get_mock_response(temperature: float, top_k: int, max_tokens: int) -> str:
    """Generate a mock response with parameter information."""
    base_response = random.choice(MOCK_RESPONSES)
    param_info = f"\n\n[Demo 参数: temperature={temperature:.1f}, top_k={top_k}, max_tokens={max_tokens}]"
    return base_response + param_info

@app.get("/")
async def root():
    """Serve the chat UI."""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/logo.svg")
async def logo():
    """Serve the NanoChat logo for favicon and header."""
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")

async def generate_stream(response_text: str) -> str:
    """Generate streaming response in legacy format."""
    # Stream each character with mock delay
    for char in response_text:
        await asyncio.sleep(0.02)  # Simulate generation delay
        chunk = {"token": char, "gpu": 0}
        yield f'data: {json.dumps(chunk, ensure_ascii=False)}\n\n'
    # Send completion marker
    yield f'data: {json.dumps({"done": True})}\n\n'

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint (streaming only) - mimics original nanochat format."""
    # Log incoming conversation to console
    logger.info("="*20)
    for message in request.messages:
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-"*20)
    
    # Generate mock response
    temperature = request.temperature if request.temperature is not None else 0.8
    top_k = request.top_k if request.top_k is not None else 50
    max_tokens = request.max_tokens if request.max_tokens is not None else 512
    response_text = get_mock_response(temperature, top_k, max_tokens)
    
    # Log the assistant response
    logger.info(f"[ASSISTANT] (Mock): {response_text}")
    logger.info("="*20)
    
    return StreamingResponse(
        generate_stream(response_text),
        media_type="text/event-stream"
    )

async def generate_openai_stream(response_text: str, request_id: str, created_at: int, model_name: str):
    """Generate OpenAI-compatible streaming response."""
    # Initial chunk with role
    initial_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created_at,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    yield f'data: {json.dumps(initial_chunk)}\n\n'
    
    # Content chunks
    for char in response_text:
        await asyncio.sleep(0.02)
        content_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_at,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": char},
                "finish_reason": None
            }]
        }
        yield f'data: {json.dumps(content_chunk)}\n\n'
    
    # Final chunk
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created_at,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f'data: {json.dumps(final_chunk)}\n\n'
    yield 'data: [DONE]\n\n'

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatRequest):
    """OpenAI-compatible chat completion endpoint (streaming & non-streaming)."""
    # Log incoming conversation to console
    logger.info("="*20)
    for message in request.messages:
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-"*20)
    
    # Generate mock response
    response_text = get_mock_response(
        request.temperature or 0.8,
        request.top_k or 50,
        request.max_tokens or 512
    )
    
    request_id = f"chatcmpl-{int(time.time())}"
    created_at = int(time.time())
    model_name = "nanochat-demo"
    
    # Log the assistant response
    logger.info(f"[ASSISTANT] (Mock): {response_text}")
    logger.info("="*20)
    
    if request.stream:
        # Streaming response
        return StreamingResponse(
            generate_openai_stream(response_text, request_id, created_at, model_name),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": created_at,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": len(response_text.split()),
                "total_tokens": 10 + len(response_text.split())
            }
        }
        return JSONResponse(content=response)

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [{
            "id": "nanochat-demo",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nanochat"
        }]
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "mode": "demo"
    }

@app.get("/stats")
async def stats():
    """Server statistics endpoint."""
    return {
        "mode": "demo",
        "workers": 1,
        "available_workers": 1
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("NanoChat Demo Server")
    print("=" * 60)
    print()
    print("Mode: DEMO (Mock responses, no model required)")
    print()
    print("Endpoints:")
    print(f"  Web UI:  http://localhost:{args.port}")
    print(f"  Legacy:  http://localhost:{args.port}/chat/completions")
    print(f"  OpenAI:  http://localhost:{args.port}/v1/chat/completions")
    print(f"  Models:  http://localhost:{args.port}/v1/models")
    print(f"  Health:  http://localhost:{args.port}/health")
    print()
    print("Features:")
    print("  - Settings panel (click gear icon)")
    print("  - Streaming responses")
    print("  - OpenAI-compatible API")
    print()
    print("Note: Responses are mock data for demonstration only")
    print("=" * 60)
    print()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )
