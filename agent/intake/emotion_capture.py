"""
Hybrid emotion detection:
  Step 1: Mental-health-focused keyword lexicon scan.
  Step 2: Ollama LLM refines + interprets the signals.

"""

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FAST_MODEL, OLLAMA_BASE_URL, EMOTION_TOP_K


MH_EMOTION_LEXICON: Dict[str, List[str]] = {
    # --- High-risk / clinical ---
    "grief": [
        r"\bloss\b", r"\bgriev\w*\b", r"\bbereavement\b", r"\bmissing\s+(him|her|them|my)\b",
        r"\bdead\b", r"\bpassed\s+away\b", r"\bgone\s+forever\b", r"\bmourning\b",
        r"\blost\s+(my|a|him|her|them)\b", r"\bmiss\s+(him|her|them|my|you)\b",
    ],
    "hopelessness": [
        r"\bno\s+hope\b", r"\bhopeless\b", r"\bnever\s+get\s+better\b",
        r"\bno\s+point\b", r"\bwhat.s\s+the\s+point\b", r"\bnothings?\s+going\s+to\s+change\b",
        r"\bgive\s+up\b", r"\bcan.t\s+(go\s+on|continue|do\s+this)\b",
    ],
    "emptiness": [
        r"\bnumb\b", r"\bempty\b", r"\bvoid\b", r"\bnothing\s+inside\b",
        r"\bfeel\s+nothing\b", r"\bhollow\b", r"\bdisconnected\b", r"\bdetach\w*\b",
    ],
    "shame": [
        r"\bsham\w*\b", r"\bembarrass\w*\b", r"\bhumiliat\w*\b",
        r"\bi.m\s+a\s+(failure|loser|burden)\b", r"\bworthless\b",
        r"\bwish\s+i\s+(was|were)\s+(different|better)\b",
    ],
    "guilt": [
        r"\bguilt\w*\b", r"\bmy\s+fault\b", r"\bi\s+(should|shouldn.t)\s+have\b",
        r"\bblame\s+myself\b", r"\bremorse\b", r"\bregret\b",
    ],
    # --- Distress emotions ---
    "sadness": [
        r"\bsad\b", r"\bcrying\b", r"\btears\b", r"\bweeping\b",
        r"\bdepressed\b", r"\bdown\b", r"\bblue\b", r"\bunhappy\b",
        r"\bheartbroken\b", r"\bhurt\b",
    ],
    "anxiety": [
        r"\banxious\b", r"\banxiety\b", r"\bworried\b", r"\bworry\b",
        r"\bpanic\w*\b", r"\bscared\b", r"\bnervous\b", r"\bfear\w*\b",
        r"\bcan.t\s+stop\s+thinking\b", r"\bowerwhelm\w*\b", r"\bunease\b",
        r"\btense\b", r"\brestless\b",
    ],
    "anger": [
        r"\bangry\b", r"\banger\b", r"\bfuriou\w*\b", r"\brage\b",
        r"\bfrustra\w*\b", r"\bresentment\b", r"\birritat\w*\b",
        r"\bhatred?\b", r"\bpissed\b",
    ],
    "loneliness": [
        r"\blonely\b", r"\bloneliness\b", r"\bisolat\w*\b", r"\balone\b",
        r"\bno\s+one\s+(cares?|understands?)\b", r"\bnobody\s+(cares?|listens?)\b",
        r"\babandoned\b",
    ],
    "overwhelm": [
        r"\boverwhelm\w*\b", r"\btoo\s+much\b", r"\bburnt?\s*out\b",
        r"\bexhaust\w*\b", r"\bdrained\b", r"\bcan.t\s+cope\b",
        r"\bstruggling\b", r"\bbreaking\s+down\b",
    ],
    # --- Mixed / neutral ---
    "confusion": [
        r"\bconfus\w*\b", r"\bdon.t\s+know\s+what\b",
        r"\bnot\s+sure\s+what\b", r"\bunsure\b", r"\bdirection\w*less\b",
    ],
    "disappointment": [
        r"\bdisappoint\w*\b", r"\blet\s+down\b", r"\bfailed\b",
        r"\bexpected\s+(more|better)\b", r"\bdid\s+not\s+(work|happen)\b",
    ],
    # --- Positive emotions ---
    "gratitude": [
        r"\bgrateful\b", r"\bthankful\b", r"\bappreciat\w*\b", r"\bblessed\b",
    ],
    "hope": [
        r"\bhopeful\b", r"\blooking\s+forward\b", r"\bgetting\s+better\b",
        r"\boptimis\w*\b", r"\bpositive\s+change\b",
    ],
    "relief": [
        r"\brelieved?\b", r"\brelief\b", r"\bfinally\b", r"\bweight\s+(lifted|off)\b",
    ],
    "joy": [
        r"\bhappy\b", r"\bjoy\w*\b", r"\bexcit\w*\b", r"\belated\b",
        r"\bgreat\b", r"\bwonderful\b",
    ],
    "neutral": [
        r"\bok\b", r"\bokay\b", r"\bfine\b", r"\balright\b",
    ],
}

EMOTION_VALENCE_MAP = {
    "grief": "negative", "hopelessness": "negative", "emptiness": "negative",
    "shame": "negative", "guilt": "negative", "sadness": "negative",
    "anxiety": "negative", "anger": "negative", "loneliness": "negative",
    "overwhelm": "negative", "confusion": "mixed", "disappointment": "negative",
    "gratitude": "positive", "hope": "positive", "relief": "positive",
    "joy": "positive", "neutral": "neutral",
}


def _scan_lexicon(text: str) -> Dict[str, float]:
  
    text_lower = text.lower()
    raw_hits: Dict[str, int] = {}
    for emotion, patterns in MH_EMOTION_LEXICON.items():
        hits = sum(1 for p in patterns if re.search(p, text_lower))
        if hits:
            raw_hits[emotion] = hits

    if not raw_hits:
        return {"neutral": 0.5}

    max_hits = max(raw_hits.values())
    return {e: round(min(h / max_hits, 1.0), 3) for e, h in raw_hits.items()}


def detect_and_interpret(text: str, llm: Optional[ChatOllama] = None) -> Dict[str, Any]:
    
    if llm is None:
        llm = ChatOllama(model=FAST_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)

    # Step 1: Fast lexicon scan
    lexicon_scores = _scan_lexicon(text)
    top_signals = sorted(lexicon_scores.items(), key=lambda x: x[1], reverse=True)[:EMOTION_TOP_K + 2]
    signals_str = ", ".join(f"{e}: {s:.2f}" for e, s in top_signals)

    # Step 2: LLM interpretation with lexicon signals as input
    system_prompt = """You are a clinical psychologist specializing in mental health emotion assessment.
You will receive a user message AND pre-computed keyword-based emotion signals.
Use BOTH to determine the most accurate emotional state.
Return ONLY valid JSON. No explanation, no markdown."""

    human_prompt = f"""User message: "{text}"

Keyword-based emotion signals (pre-computed): {signals_str}

Using the signals as strong hints and the full message for context, return JSON:
{{
  "primary_emotion": "string (dominant emotion — trust the signals unless context clearly overrides)",
  "secondary_emotions": ["up to 3 other emotions present"],
  "intensity": float (0.0-10.0, 10=most intense),
  "valence": "positive|negative|mixed|neutral",
  "therapeutic_need": "validation|agency|meaning|safety|exploration",
  "brief_assessment": "1 concise clinical sentence"
}}

therapeutic_need guide:
- validation: user needs to feel heard and understood
- agency: user needs to feel in control / empowered
- meaning: user is searching for purpose or reason
- safety: user is in distress or crisis — needs grounding
- exploration: user wants to understand themselves better"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        result.setdefault("hf_scores", lexicon_scores)   # store lexicon scores for transparency
        return result

    except Exception as e:
        print(f"[EmotionDetector] LLM interpretation error: {e}")
        # Fallback directly from lexicon — still better than plain "neutral"
        top = top_signals[0] if top_signals else ("neutral", 0.5)
        return {
            "primary_emotion": top[0],
            "secondary_emotions": [e for e, _ in top_signals[1:3]],
            "intensity": round(top[1] * 10, 1),
            "valence": EMOTION_VALENCE_MAP.get(top[0], "neutral"),
            "therapeutic_need": "validation",
            "brief_assessment": f"User appears to be experiencing {top[0]}.",
            "hf_scores": lexicon_scores,
        }
