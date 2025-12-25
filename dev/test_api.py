"""Test safety data generation"""
import sys
sys.path.insert(0, '.')

from dev.gen_safety_data import generate_conversation

print("Testing safety conversation generation...")
messages, category = generate_conversation(0)

print(f"Category: {category}")
print(f"Messages: {len(messages)} turns")
print("-" * 50)
for m in messages:
    role = m["role"]
    content = m["content"]
    print(f"[{role.upper()}]: {content[:200]}...")
    print()
