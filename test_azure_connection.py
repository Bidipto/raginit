#!/usr/bin/env python3

import os
from dotenv import load_dotenv

load_dotenv()

print("=== Environment Variables Check ===")
print(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
print(
    f"AZURE_OPENAI_API_KEY: {os.getenv('AZURE_OPENAI_API_KEY')[:10]}..."
    if os.getenv("AZURE_OPENAI_API_KEY")
    else "Not set"
)
print(f"AZURE_OPENAI_API_VERSION: {os.getenv('AZURE_OPENAI_API_VERSION')}")
print(
    f"AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME')}"
)
print(
    f"AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: {os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')}"
)

print("\n=== Testing Azure OpenAI Client ===")

try:
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    print("✅ Azure OpenAI client created successfully!")

    # Test a simple embedding request
    print("\n=== Testing Embeddings ===")
    response = client.embeddings.create(
        input=["Hello world"], model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    )
    print(f"✅ Embedding test successful! Dimension: {len(response.data[0].embedding)}")

    # Test a simple chat request
    print("\n=== Testing Chat ===")
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        messages=[{"role": "user", "content": "Say hello!"}],
        max_tokens=50,
    )
    print(f"✅ Chat test successful! Response: {response.choices[0].message.content}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
