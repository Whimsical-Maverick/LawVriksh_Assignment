import os
from typing import Dict, Any, Tuple
import tiktoken

# Optional OpenAI client
from openai import OpenAI

MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def count_tokens(text: str, model: str = "gpt-3.5-turbo"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))

def llm_complete_json(prompt: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Returns (parsed_json, token_usage). If OPENAI_API_KEY absent, returns a heuristic dummy.
    """
    prompt_tokens = count_tokens(prompt, model=MODEL)

    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content
        usage = resp.usage
        comp_tokens = usage.completion_tokens if usage else count_tokens(text, model=MODEL)
        return (safe_json(text), {"prompt": prompt_tokens, "completion": comp_tokens, "total": prompt_tokens + comp_tokens})

    # Fallback heuristic (no network): return empty/dummy JSON and estimate tokens
    dummy = {"suggestions": [], "weak_sections": [], "readability": None, "relevance": None, "final_score": None}
    completion_tokens = max(24, int(prompt_tokens * 0.2))
    return dummy, {"prompt": prompt_tokens, "completion": completion_tokens, "total": prompt_tokens + completion_tokens}

def safe_json(text: str):
    import json
    try:
        return json.loads(text)
    except Exception:
        return {}
