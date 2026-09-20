from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-20b"

def generate_response(message: list[dict], max_tokens: int = 500) -> str: 
    completion = client.chat.completions.create(
        messages=message, 
        model=MODEL, 
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content