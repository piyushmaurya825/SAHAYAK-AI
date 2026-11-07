"""
Extracts core values and value conflicts from user message.
"""

from __future__ import annotations
import json
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FAST_MODEL, OLLAMA_BASE_URL

def extract_values(user_message: str, llm: Optional[ChatOllama] = None) -> dict:
    if llm is None:
        llm = ChatOllama(model=FAST_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)

    system = "You are a values-based psychologist. Identify core moral values and conflicts. Return ONLY valid JSON."
    human = f"""User message: "{user_message}"

Return JSON:
{{
  "core_values": ["list of values like duty, family_honor, authenticity, achievement, compassion"],
  "central_conflict": "The main value tension or empty string if none",
  "cultural_value_pressures": ["external values being imposed"],
  "personal_values": ["user own internalized values"]
}}"""

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        content = resp.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"): content = content[4:]
        return json.loads(content)
    except Exception as e:
        print(f"[ValuesExtractor] Error: {e}")
        return {"core_values": [], "central_conflict": "", "cultural_value_pressures": [], "personal_values": []}
