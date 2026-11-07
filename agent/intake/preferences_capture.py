"""
Detects user counselling preferences (tone, modality, language style).
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

HINGLISH_MARKERS = ["yaar", "bhai", "kya", "nahi", "hai", "matlab", "bas", "accha", "toh"]

def detect_preferences(user_message: str, existing_prefs: dict = None, llm: Optional[ChatOllama] = None) -> dict:
    msg_lower = user_message.lower()
    
    # Rule-based language detection
    hinglish_count = sum(1 for w in HINGLISH_MARKERS if w in msg_lower)
    lang_style = "Hinglish" if hinglish_count >= 2 else "formal English"

    wants_practical = any(w in msg_lower for w in ["how to", "steps", "what should i", "help me", "solution"])
    wants_spiritual = any(w in msg_lower for w in ["gita", "spiritual", "god", "meaning", "purpose", "philosophy"])
    just_listening = any(w in msg_lower for w in ["just want to talk", "listen", "vent", "share"])

    if existing_prefs:
        lang_style = existing_prefs.get("language_style", lang_style)

    return {
        "tone": "gentle" if just_listening else "empathetic",
        "modality": "practical" if wants_practical else ("spiritual" if wants_spiritual else "balanced"),
        "language_style": lang_style,
        "wants_practical_steps": wants_practical,
        "wants_spiritual_guidance": wants_spiritual or True,
        "just_wants_to_be_heard": just_listening,
    }
