import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

token = os.getenv("GITHUB_TOKEN")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=token,
)

try:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello, world!"}],
        model="gpt-4o",
    )
    print("✅ Success! GitHub Models Access is working properly.")
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print("❌ Connection Failed:")
    print(e)