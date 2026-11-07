"""
Extracts cultural context from user message using LLM structured output.
"""

from __future__ import annotations
import json
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FAST_MODEL, OLLAMA_BASE_URL, SUPPORTED_REGIONS

def extract_cultural_background(
    user_message: str,
    conversation_history: list = None,
    existing_profile: dict = None,
    llm: Optional[ChatOllama] = None
) -> dict:
    """Extract cultural background from user message and existing profile."""
    if llm is None:
        llm = ChatOllama(model=FAST_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)

    history_str = ""
    if conversation_history:
        history_str = "\n".join(
            f"{m.get('role','user').upper()}: {m.get('content','')[:200]}"
            for m in conversation_history[-4:]
        )

    profile_str = json.dumps(existing_profile) if existing_profile else "No prior profile."

    system = """You are a cultural intelligence analyst. Analyze the user message to infer cultural background.
Return ONLY valid JSON. No markdown, no explanation."""

    human = f"""User message: "{user_message}"
Recent conversation: {history_str or "None"}
Existing profile: {profile_str}

Return JSON:
{{
  "region": "one of: {', '.join(SUPPORTED_REGIONS)}",
  "community": "specific community or Unknown",
  "family_structure": "collectivist|individualist|unknown",
  "collectivism_index": float 0.0-1.0,
  "cultural_tags": ["list", "of", "tags", "like", "joint_family_dynamics", "academic_pressure"],
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
        print(f"[CulturalExtractor] Error: {e}")
        return {
            "region": "Universal", "community": "Unknown",
            "family_structure": "unknown", "collectivism_index": 0.5,
            "cultural_tags": [], "confidence": 0.0
        }
