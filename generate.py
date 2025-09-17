from __future__ import annotations
import requests
from config import CONFIG

def ollama_generate(prompt: str) -> str:
    url = f"{CONFIG['OLLAMA_BASE_URL']}/api/generate"
    payload = {
        "model": CONFIG["LLM_MODEL"],
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": CONFIG["TEMPERATURE"]}
    }
    r = requests.post(url, json=payload, timeout=CONFIG["REQUEST_TIMEOUT"]).json()
    return r.get("response", "")

IDEA_PROMPT = """
Whatever your prompt is
"""

OUTLINE_PROMPT = """
same as above
"""

EXPAND_PROMPT = """
same as above
"""

DRAFT_PROMPT = """
same as above
"""

def ideas(beat: str, constraints: str) -> str:
    prompt = f"{IDEA_PROMPT}\nBeat: {beat}\nConstraints: {constraints}\n"
    return ollama_generate(prompt)

def outline(title: str, context: str) -> str:
    prompt = f"{OUTLINE_PROMPT}\nWorking title: {title}\nContext bullets:\n{context}\n"
    return ollama_generate(prompt)

def expand(outline_text: str, retrieved_snippets: str) -> str:
    prompt = f"{EXPAND_PROMPT}\nOutline:\n{outline_text}\n\nRetrieved context (use for citations):\n{retrieved_snippets}\n"
    return ollama_generate(prompt)

def draft(title: str, notes: str, retrieved_snippets: str) -> str:
    prompt = f"{DRAFT_PROMPT}\nTitle: {title}\nReporter notes:\n{notes}\n\nRetrieved context (for [n] citations):\n{retrieved_snippets}\n"
    return ollama_generate(prompt)
