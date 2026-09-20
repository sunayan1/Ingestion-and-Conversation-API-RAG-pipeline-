import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def get_history(session_id: str) -> list[dict]: 
    key = f"chat_history:{session_id}"
    raw = r.get(key)

    if raw is None: 
        return []

    return json.loads(raw)

def save_history(session_id: str, history: list[dict]) -> None:
    key = f"chat_history:{session_id}"
    r.set(key, json.dumps(history))