"""
Detects religious/spiritual beliefs from user message.
"""

from __future__ import annotations
import json
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FAST_MODEL, OLLAMA_BASE_URL, SUPPORTED_FAITHS

FAITH_KEYWORDS = {
    "Hindu": ["hindu", "gita", "krishna", "dharma", "karma", "puja", "mandir", "vedic"],
    "Muslim": ["muslim", "islam", "allah", "quran", "prayer", "namaz", "mosque", "ramadan"],
    "Christian": ["christian", "jesus", "church", "bible", "god", "prayer", "christ"],
    "Buddhist": ["buddhist", "buddha", "meditation", "dharma", "nirvana", "sangha"],
    "Sikh": ["sikh", "guru", "gurdwara", "waheguru", "seva"],
    "Atheist": ["atheist", "no religion", "don't believe", "no god", "secular"],
    "Agnostic": ["agnostic", "not sure about god", "spiritual but not religious"],
}

def detect_faith(user_message: str, llm: Optional[ChatOllama] = None) -> dict:
    msg_lower = user_message.lower()
    detected_faith = "Spiritual (non-religious)"
    for faith, keywords in FAITH_KEYWORDS.items():
        if any(k in msg_lower for k in keywords):
            detected_faith = faith
            break

    if llm is None:
        llm = ChatOllama(model=FAST_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)

    system = "You are a compassionate spiritual counselor. Analyze the user message for religious/spiritual identity. Return ONLY valid JSON."
    human = f"""User message: "{user_message}"
Pre-detected faith hint: {detected_faith}

Return JSON:
{{
  "faith": "one of: {', '.join(SUPPORTED_FAITHS)}",
  "openness_to_spiritual_counselling": true or false,
  "spirituality_level": "strict|moderate|casual|secular",
  "wants_philosophical_framing": true or false,
  "confidence": float 0.0-1.0
}}"""

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        content = resp.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"): content = content[4:]
        return json.loads(content)
    except Exception as e:
        print(f"[FaithDetector] Error: {e}")
        return {
            "faith": detected_faith,
            "openness_to_spiritual_counselling": True,
            "spirituality_level": "moderate",
            "wants_philosophical_framing": True,
            "confidence": 0.3
        }
