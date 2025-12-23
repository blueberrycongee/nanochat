# Nanochat API Documentation

This document describes the OpenAI-compatible API endpoints added to Nanochat.

## Overview

Nanochat now provides an OpenAI-compatible API that allows you to use Nanochat as a drop-in replacement for OpenAI's chat completions API. This enables integration with existing tools and libraries that support the OpenAI API format.

## Features

- **OpenAI-compatible `/v1/chat/completions` endpoint**
- **Streaming and non-streaming responses**
- **Rate limiting** (60 requests per minute per client)
- **Request validation and abuse prevention**
- **Model listing via `/v1/models`**
- **Health check endpoint**

## Endpoints

### POST /v1/chat/completions

Create a chat completion (OpenAI-compatible).

**Request Body:**

```json
{
  "model": "nanochat",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.8,
  "max_tokens": 512,
  "top_k": 50,
  "stream": false,
  "stop": ["\n\n"]
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | "nanochat" | Model identifier (ignored, uses loaded model) |
| `messages` | array | required | Array of message objects |
| `temperature` | float | 0.8 | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | 512 | Maximum tokens to generate (1-4096) |
| `top_k` | int | 50 | Top-k sampling parameter (1-200) |
| `top_p` | float | - | Nucleus sampling (not yet implemented) |
| `stream` | bool | false | Enable streaming responses |
| `stop` | string/array | null | Stop sequences |
| `n` | int | 1 | Number of completions (1-8) |
| `user` | string | null | User identifier for rate limiting |

**Response (non-streaming):**

```json
{
  "id": "chatcmpl-abc123...",
  "object": "chat.completion",
  "created": 1703318400,
  "model": "nanochat-sft",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20
  }
}
```

**Response (streaming):**

When `stream: true`, responses are sent as Server-Sent Events:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1703318400,"model":"nanochat-sft","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1703318400,"model":"nanochat-sft","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1703318400,"model":"nanochat-sft","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### GET /v1/models

List available models.

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "nanochat-sft",
      "object": "model",
      "created": 1703318400,
      "owned_by": "nanochat"
    }
  ]
}
```

### GET /v1/models/{model_id}

Get details of a specific model.

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "ok",
  "ready": true,
  "num_gpus": 1,
  "available_workers": 1
}
```

### GET /stats

Worker pool statistics.

**Response:**

```json
{
  "total_workers": 1,
  "available_workers": 1,
  "busy_workers": 0,
  "workers": [
    {"gpu_id": 0, "device": "cuda:0"}
  ]
}
```

## Usage Examples

### cURL

**Non-streaming request:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nanochat",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

**Streaming request:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nanochat",
    "messages": [
      {"role": "user", "content": "Tell me a short story."}
    ],
    "stream": true
  }'
```

### Python (OpenAI Client)

```python
from openai import OpenAI

# Point to local Nanochat server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Nanochat doesn't require API key
)

# Non-streaming
response = client.chat.completions.create(
    model="nanochat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.8,
    max_tokens=512
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="nanochat",
    messages=[
        {"role": "user", "content": "Tell me a joke."}
    ],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### Python (requests)

```python
import requests
import json

# Non-streaming
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "nanochat",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.8
    }
)
print(response.json()["choices"][0]["message"]["content"])

# Streaming
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "nanochat",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: ') and not line.startswith('data: [DONE]'):
            data = json.loads(line[6:])
            if data['choices'][0]['delta'].get('content'):
                print(data['choices'][0]['delta']['content'], end='', flush=True)
print()
```

### JavaScript/TypeScript

```typescript
// Using fetch API
async function chat(message: string): Promise<string> {
  const response = await fetch('http://localhost:8000/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'nanochat',
      messages: [{ role: 'user', content: message }],
      temperature: 0.8
    })
  });
  
  const data = await response.json();
  return data.choices[0].message.content;
}

// Streaming with EventSource-like handling
async function* streamChat(message: string) {
  const response = await fetch('http://localhost:8000/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'nanochat',
      messages: [{ role: 'user', content: message }],
      stream: true
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');
    
    for (const line of lines) {
      if (line.startsWith('data: ') && !line.includes('[DONE]')) {
        const data = JSON.parse(line.slice(6));
        if (data.choices[0].delta.content) {
          yield data.choices[0].delta.content;
        }
      }
    }
  }
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **60 requests per minute** per client IP
- Exceeded limits return HTTP 429 with `Retry-After` header

## Error Handling

The API returns standard HTTP error codes:

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model not ready |

Error responses include a `detail` field with description:

```json
{
  "detail": "Rate limit exceeded. Please try again later."
}
```

## Compatibility Notes

- **System messages**: Nanochat doesn't have a native system role. System messages are prepended to user messages with `[System]:` prefix.
- **Function calling**: Not supported.
- **Vision/Images**: Not supported.
- **top_p**: Parameter accepted but not yet implemented.
- **n > 1**: Multiple completions in a single request not fully implemented.

## Web UI Settings

The web UI now includes a settings panel (click the gear icon) with sliders for:

- **Temperature** (0.0-2.0): Controls randomness
- **Top-K** (1-200): Number of top tokens to consider
- **Max Tokens** (64-2048): Maximum response length

Settings take effect immediately for the next request.
