#!/usr/bin/env python3
"""
Unified web chat server - serves both UI and API from a single FastAPI instance.

Uses data parallelism to distribute requests across multiple GPUs. Each GPU loads
a full copy of the model, and incoming requests are distributed to available workers.

Launch examples:

- single available GPU (default)
python -m scripts.chat_web

- 4 GPUs
python -m scripts.chat_web --num-gpus 4

To chat, open the URL printed in the console. (If on cloud box, make sure to use public IP)

Endpoints:
  GET  /                    - Chat UI
  POST /chat/completions    - Chat API (legacy, streaming only)
  POST /v1/chat/completions - OpenAI-compatible Chat API (streaming & non-streaming)
  GET  /v1/models           - List available models (OpenAI-compatible)
  GET  /health              - Health check with worker pool status
  GET  /stats               - Worker pool statistics and GPU utilization

Abuse Prevention:
  - Maximum 500 messages per request
  - Maximum 8000 characters per message
  - Maximum 32000 characters total conversation length
  - Temperature clamped to 0.0-2.0
  - Top-k clamped to 1-200
  - Top-p clamped to 0.0-1.0
  - Max tokens clamped to 1-4096
  - Rate limiting: 60 requests per minute per client
"""

import argparse
import json
import os
import torch
import asyncio
import logging
import random
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, AsyncGenerator, Union, Dict, Any
from dataclasses import dataclass
from contextlib import nullcontext
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 1
MAX_TOP_K = 200
MIN_TOP_P = 0.0
MAX_TOP_P = 1.0
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

# Rate limiting settings
RATE_LIMIT_REQUESTS = 60  # requests per window
RATE_LIMIT_WINDOW = 60    # window in seconds

parser = argparse.ArgumentParser(description='NanoChat Web Server')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Default temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Default max tokens for generation')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
args = parser.parse_args()

# Configure logging for conversation traffic
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

@dataclass
class Worker:
    """A worker with a model loaded on a specific GPU."""
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object
    autocast_ctx: torch.amp.autocast

class WorkerPool:
    """Pool of workers, each with a model replica on a different GPU."""

    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1 # e.g. cpu|mps
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """Load model on each GPU."""
        print(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "Only CUDA supports multiple workers/GPUs. cpu|mps does not."

        for gpu_id in range(self.num_gpus):

            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(device_type) # e.g. cpu|mps
                print(f"Loading model on {device_type}...")

            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
                autocast_ctx=autocast_ctx
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        """Get an available worker from the pool."""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """Return a worker to the pool."""
        await self.available_workers.put(worker)


# -----------------------------------------------------------------------------
# Rate Limiter
# -----------------------------------------------------------------------------
class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""
    
    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> bool:
        """Check if a request from client_id is allowed."""
        async with self._lock:
            now = time.time()
            # Clean old requests outside the window
            self.requests[client_id] = [
                t for t in self.requests[client_id] 
                if now - t < self.window_seconds
            ]
            # Check if under limit
            if len(self.requests[client_id]) < self.max_requests:
                self.requests[client_id].append(now)
                return True
            return False
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        now = time.time()
        recent = [t for t in self.requests.get(client_id, []) if now - t < self.window_seconds]
        return max(0, self.max_requests - len(recent))


# -----------------------------------------------------------------------------
# Legacy API Models (for backward compatibility)
# -----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None


# -----------------------------------------------------------------------------
# OpenAI-Compatible API Models
# -----------------------------------------------------------------------------
class OpenAIMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = "nanochat"
    messages: List[OpenAIMessage]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=8)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None
    # Extension: top_k support (not in OpenAI API but useful)
    top_k: Optional[int] = Field(default=None, ge=1, le=200)

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str

class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIChoiceMessage
    finish_reason: str

class OpenAIChatResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage

class OpenAIStreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class OpenAIStreamChoice(BaseModel):
    index: int
    delta: OpenAIStreamDelta
    finish_reason: Optional[str] = None

class OpenAIStreamChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIStreamChoice]

class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class OpenAIModelList(BaseModel):
    object: str = "list"
    data: List[OpenAIModel]


def validate_chat_request(request: ChatRequest):
    """Validate chat request to prevent abuse."""
    # Check number of messages
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request"
        )

    # Check individual message lengths and total conversation length
    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message"
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed"
        )

    # Validate role values (legacy API only supports user and assistant)
    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role '{message.role}'. Must be 'user' or 'assistant'. Use /v1/chat/completions for system message support."
            )

    # Validate temperature
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
            )

    # Validate top_k
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}"
            )

    # Validate max_tokens
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}"
            )


def validate_openai_request(request: OpenAIChatRequest):
    """Validate OpenAI-compatible chat request."""
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} allowed"
        )
    
    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")
        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} exceeds {MAX_MESSAGE_LENGTH} character limit"
            )
        total_length += msg_length
    
    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation exceeds {MAX_TOTAL_CONVERSATION_LENGTH} character limit"
        )
    
    # Validate role values (system is allowed in OpenAI format)
    valid_roles = {"user", "assistant", "system"}
    for i, message in enumerate(request.messages):
        if message.role not in valid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role '{message.role}'. Must be one of: {valid_roles}"
            )


def get_client_ip(request: Request) -> str:
    """Extract client IP for rate limiting."""
    # Check for forwarded headers (behind proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# Global rate limiter instance
rate_limiter = RateLimiter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on all GPUs on startup."""
    print("Loading nanochat models across GPUs...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    app.state.rate_limiter = rate_limiter
    app.state.model_name = f"nanochat-{args.source}"
    app.state.created_at = int(time.time())
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    print(f"Server ready at http://localhost:{args.port}")
    print(f"OpenAI-compatible API available at http://localhost:{args.port}/v1/chat/completions")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Serve the chat UI."""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # Replace the API_URL to use the same origin
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    """Serve the NanoChat logo for favicon and header."""
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")

async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """Generate assistant response with streaming."""
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    # Accumulate tokens to properly handle multi-byte UTF-8 characters (like emojis)
    accumulated_tokens = []
    # Track the last complete UTF-8 string (without replacement characters)
    last_clean_text = ""

    with worker.autocast_ctx:
        for token_column, token_masks in worker.engine.generate(
            tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=random.randint(0, 2**31 - 1)
        ):
            token = token_column[0]

            # Stopping criteria
            if token == assistant_end or token == bos:
                break

            # Append the token to sequence
            accumulated_tokens.append(token)
            # Decode all accumulated tokens to get proper UTF-8 handling
            # Note that decode is a quite efficient operation, basically table lookup and string concat
            current_text = worker.tokenizer.decode(accumulated_tokens)
            # Only emit text if it doesn't end with a replacement character
            # This ensures we don't emit incomplete UTF-8 sequences
            if not current_text.endswith('�'):
                # Extract only the new text since last clean decode
                new_text = current_text[len(last_clean_text):]
                if new_text:  # Only yield if there's new content
                    yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                    last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint (streaming only) - uses worker pool for multi-GPU."""

    # Basic validation to prevent abuse
    validate_chat_request(request)

    # Log incoming conversation to console
    logger.info("="*20)
    for i, message in enumerate(request.messages):
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-"*20)

    # Acquire a worker from the pool (will wait if all are busy)
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        # Build conversation tokens
        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<|user_start|>")
        user_end = worker.tokenizer.encode_special("<|user_end|>")
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

        conversation_tokens = [bos]
        for message in request.messages:
            if message.role == "user":
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)

        conversation_tokens.append(assistant_start)

        # Streaming response with worker release after completion
        response_tokens = []
        async def stream_and_release():
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k
                ):
                    # Accumulate response for logging
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    yield chunk
            finally:
                # Log the assistant response to console
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                logger.info("="*20)
                # Release worker back to pool after streaming is done
                await worker_pool.release_worker(worker)

        return StreamingResponse(
            stream_and_release(),
            media_type="text/event-stream"
        )
    except Exception as e:
        # Make sure to release worker even on error
        await worker_pool.release_worker(worker)
        raise e

@app.get("/health")
async def health():
    """Health check endpoint."""
    worker_pool = getattr(app.state, 'worker_pool', None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0
    }

@app.get("/stats")
async def stats():
    """Get worker pool statistics."""
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {
                "gpu_id": w.gpu_id,
                "device": str(w.device)
            } for w in worker_pool.workers
        ]
    }


# -----------------------------------------------------------------------------
# OpenAI-Compatible API Endpoints
# -----------------------------------------------------------------------------

def build_conversation_tokens(worker: Worker, messages: List[OpenAIMessage]) -> tuple:
    """Build conversation tokens from OpenAI-style messages. Returns (tokens, prompt_token_count)."""
    bos = worker.tokenizer.get_bos_token_id()
    user_start = worker.tokenizer.encode_special("<|user_start|>")
    user_end = worker.tokenizer.encode_special("<|user_end|>")
    assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    
    conversation_tokens = [bos]
    for message in messages:
        if message.role == "system":
            # Treat system messages as user messages (nanochat doesn't have system role)
            conversation_tokens.append(user_start)
            conversation_tokens.extend(worker.tokenizer.encode(f"[System]: {message.content}"))
            conversation_tokens.append(user_end)
        elif message.role == "user":
            conversation_tokens.append(user_start)
            conversation_tokens.extend(worker.tokenizer.encode(message.content))
            conversation_tokens.append(user_end)
        elif message.role == "assistant":
            conversation_tokens.append(assistant_start)
            conversation_tokens.extend(worker.tokenizer.encode(message.content))
            conversation_tokens.append(assistant_end)
    
    conversation_tokens.append(assistant_start)
    return conversation_tokens, len(conversation_tokens)


async def generate_openai_stream(
    worker: Worker,
    tokens: List[int],
    request_id: str,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    top_k: Optional[int],
    stop_sequences: Optional[List[str]] = None
) -> AsyncGenerator[str, None]:
    """Generate OpenAI-compatible streaming response."""
    created = int(time.time())
    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()
    
    accumulated_tokens = []
    last_clean_text = ""
    full_response = ""
    
    # Send initial chunk with role
    initial_chunk = OpenAIStreamChunk(
        id=request_id,
        created=created,
        model=model_name,
        choices=[OpenAIStreamChoice(
            index=0,
            delta=OpenAIStreamDelta(role="assistant"),
            finish_reason=None
        )]
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"
    
    finish_reason = "stop"
    with worker.autocast_ctx:
        for token_column, token_masks in worker.engine.generate(
            tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=random.randint(0, 2**31 - 1)
        ):
            token = token_column[0]
            
            if token == assistant_end or token == bos:
                break
            
            accumulated_tokens.append(token)
            current_text = worker.tokenizer.decode(accumulated_tokens)
            
            if not current_text.endswith('�'):
                new_text = current_text[len(last_clean_text):]
                if new_text:
                    full_response += new_text
                    
                    # Check for stop sequences
                    if stop_sequences:
                        for stop_seq in stop_sequences:
                            if stop_seq in full_response:
                                # Truncate at stop sequence
                                idx = full_response.find(stop_seq)
                                new_text = new_text[:max(0, len(new_text) - (len(full_response) - idx))]
                                finish_reason = "stop"
                                if new_text:
                                    chunk = OpenAIStreamChunk(
                                        id=request_id,
                                        created=created,
                                        model=model_name,
                                        choices=[OpenAIStreamChoice(
                                            index=0,
                                            delta=OpenAIStreamDelta(content=new_text),
                                            finish_reason=None
                                        )]
                                    )
                                    yield f"data: {chunk.model_dump_json()}\n\n"
                                break
                        else:
                            # No stop sequence found, continue normally
                            chunk = OpenAIStreamChunk(
                                id=request_id,
                                created=created,
                                model=model_name,
                                choices=[OpenAIStreamChoice(
                                    index=0,
                                    delta=OpenAIStreamDelta(content=new_text),
                                    finish_reason=None
                                )]
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"
                            last_clean_text = current_text
                            continue
                        break  # Stop sequence found
                    else:
                        chunk = OpenAIStreamChunk(
                            id=request_id,
                            created=created,
                            model=model_name,
                            choices=[OpenAIStreamChoice(
                                index=0,
                                delta=OpenAIStreamDelta(content=new_text),
                                finish_reason=None
                            )]
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    last_clean_text = current_text
        else:
            # Loop completed without break (max tokens reached)
            finish_reason = "length"
    
    # Send final chunk with finish_reason
    final_chunk = OpenAIStreamChunk(
        id=request_id,
        created=created,
        model=model_name,
        choices=[OpenAIStreamChoice(
            index=0,
            delta=OpenAIStreamDelta(),
            finish_reason=finish_reason
        )]
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


async def generate_openai_complete(
    worker: Worker,
    tokens: List[int],
    prompt_tokens: int,
    request_id: str,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    top_k: Optional[int],
    stop_sequences: Optional[List[str]] = None
) -> OpenAIChatResponse:
    """Generate complete (non-streaming) OpenAI-compatible response."""
    created = int(time.time())
    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()
    
    accumulated_tokens = []
    full_response = ""
    finish_reason = "stop"
    
    with worker.autocast_ctx:
        for token_column, token_masks in worker.engine.generate(
            tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=random.randint(0, 2**31 - 1)
        ):
            token = token_column[0]
            
            if token == assistant_end or token == bos:
                break
            
            accumulated_tokens.append(token)
        else:
            finish_reason = "length"
    
    full_response = worker.tokenizer.decode(accumulated_tokens)
    
    # Check for stop sequences and truncate
    if stop_sequences:
        for stop_seq in stop_sequences:
            if stop_seq in full_response:
                full_response = full_response[:full_response.find(stop_seq)]
                finish_reason = "stop"
                break
    
    completion_tokens = len(accumulated_tokens)
    
    return OpenAIChatResponse(
        id=request_id,
        created=created,
        model=model_name,
        choices=[OpenAIChoice(
            index=0,
            message=OpenAIChoiceMessage(content=full_response),
            finish_reason=finish_reason
        )],
        usage=OpenAIUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatRequest, http_request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    
    Supports both streaming (stream=true) and non-streaming responses.
    Compatible with OpenAI Python client and other OpenAI-compatible tools.
    """
    # Rate limiting
    client_ip = get_client_ip(http_request)
    if not await rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW)}
        )
    
    # Validate request
    validate_openai_request(request)
    
    # Log incoming request
    logger.info("="*20 + " [OpenAI API] " + "="*20)
    for message in request.messages:
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-"*20)
    
    # Acquire worker
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()
    
    try:
        # Build conversation tokens
        conversation_tokens, prompt_tokens = build_conversation_tokens(worker, request.messages)
        
        # Prepare parameters
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        model_name = app.state.model_name
        temperature = request.temperature if request.temperature is not None else args.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else args.max_tokens
        top_k = request.top_k if request.top_k is not None else args.top_k
        
        # Handle stop sequences
        stop_sequences = None
        if request.stop:
            stop_sequences = [request.stop] if isinstance(request.stop, str) else request.stop
        
        if request.stream:
            # Streaming response
            async def stream_and_release():
                try:
                    full_response_parts = []
                    async for chunk in generate_openai_stream(
                        worker, conversation_tokens, request_id, model_name,
                        temperature, max_tokens, top_k, stop_sequences
                    ):
                        # Try to extract content for logging
                        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                            try:
                                data = json.loads(chunk[6:])
                                if "choices" in data and data["choices"]:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta and delta["content"]:
                                        full_response_parts.append(delta["content"])
                            except:
                                pass
                        yield chunk
                finally:
                    full_response = "".join(full_response_parts)
                    logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                    logger.info("="*50)
                    await worker_pool.release_worker(worker)
            
            return StreamingResponse(
                stream_and_release(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming response
            response = await generate_openai_complete(
                worker, conversation_tokens, prompt_tokens, request_id, model_name,
                temperature, max_tokens, top_k, stop_sequences
            )
            logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {response.choices[0].message.content}")
            logger.info("="*50)
            await worker_pool.release_worker(worker)
            return response
    
    except Exception as e:
        await worker_pool.release_worker(worker)
        logger.error(f"Error in OpenAI chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    model_name = getattr(app.state, 'model_name', 'nanochat')
    created_at = getattr(app.state, 'created_at', int(time.time()))
    
    return OpenAIModelList(
        data=[
            OpenAIModel(
                id=model_name,
                created=created_at,
                owned_by="nanochat"
            )
        ]
    )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get details of a specific model."""
    model_name = getattr(app.state, 'model_name', 'nanochat')
    created_at = getattr(app.state, 'created_at', int(time.time()))
    
    if model_id != model_name and model_id != "nanochat":
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    return OpenAIModel(
        id=model_name,
        created=created_at,
        owned_by="nanochat"
    )


if __name__ == "__main__":
    import uvicorn
    print(f"Starting NanoChat Web Server")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    uvicorn.run(app, host=args.host, port=args.port)
