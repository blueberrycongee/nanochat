#!/usr/bin/env python3
"""
Example client for Nanochat OpenAI-compatible API.

Usage:
    pip install openai
    python examples/openai_client_example.py

Make sure the Nanochat server is running:
    python -m scripts.chat_web
"""

from openai import OpenAI

# Configuration
NANOCHAT_URL = "http://localhost:8000/v1"

def main():
    # Create client pointing to Nanochat
    client = OpenAI(
        base_url=NANOCHAT_URL,
        api_key="not-needed"  # Nanochat doesn't require API key
    )
    
    print("=" * 50)
    print("Nanochat OpenAI-Compatible API Demo")
    print("=" * 50)
    
    # Example 1: List available models
    print("\n1. Listing available models...")
    models = client.models.list()
    for model in models.data:
        print(f"   - {model.id} (owned by: {model.owned_by})")
    
    # Example 2: Simple chat completion (non-streaming)
    print("\n2. Non-streaming chat completion...")
    response = client.chat.completions.create(
        model="nanochat",
        messages=[
            {"role": "system", "content": "You are a helpful and friendly assistant."},
            {"role": "user", "content": "What is 2 + 2?"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    print(f"   Response: {response.choices[0].message.content}")
    print(f"   Tokens used: {response.usage.total_tokens}")
    
    # Example 3: Streaming chat completion
    print("\n3. Streaming chat completion...")
    print("   Response: ", end="", flush=True)
    stream = client.chat.completions.create(
        model="nanochat",
        messages=[
            {"role": "user", "content": "Write a haiku about coding."}
        ],
        stream=True,
        temperature=0.9,
        max_tokens=100
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")
    
    # Example 4: Multi-turn conversation
    print("4. Multi-turn conversation...")
    conversation = [
        {"role": "user", "content": "My name is Alice."},
    ]
    
    response = client.chat.completions.create(
        model="nanochat",
        messages=conversation,
        temperature=0.8,
        max_tokens=100
    )
    assistant_reply = response.choices[0].message.content
    print(f"   User: My name is Alice.")
    print(f"   Assistant: {assistant_reply}")
    
    # Continue conversation
    conversation.append({"role": "assistant", "content": assistant_reply})
    conversation.append({"role": "user", "content": "What's my name?"})
    
    response = client.chat.completions.create(
        model="nanochat",
        messages=conversation,
        temperature=0.8,
        max_tokens=100
    )
    print(f"   User: What's my name?")
    print(f"   Assistant: {response.choices[0].message.content}")
    
    print("\n" + "=" * 50)
    print("Demo complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
