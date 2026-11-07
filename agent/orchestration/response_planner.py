"""
Creates a 4-step pedagogical response plan before prose generation.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PRIMARY_MODEL, OLLAMA_BASE_URL, GENERATION_TEMPERATURE

THERAPEUTIC_GOAL_MAP = {
    "validation": "Provide deep emotional validation and normalized empathy",
    "agency": "Restore sense of personal control and self-efficacy",
    "meaning": "Help user find philosophical/spiritual meaning in situation",
    "safety": "De-escalate distress and stabilize emotional state",
    "exploration": "Open reflective dialogue to deepen self-understanding",
}

def create_response_plan(
    user_message: str,
    user_package: Dict[str, Any],
    memory_context: Optional[str],
    assembled_context: str,
    llm: Optional[ChatOllama] = None
) -> Dict[str, Any]:
    """Generate a structured therapeutic response plan."""
    if llm is None:
        llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)

    user_package = user_package or {}
    emotions = user_package.get("emotions", {}) or {}
    cultural = user_package.get("cultural", {}) or {}
    faith = user_package.get("faith", {}) or {}
    prefs = user_package.get("preferences", {}) or {}
    values = user_package.get("values", {}) or {}

    memory_str = f"Memory note: {memory_context}" if memory_context else "First session with this user."

    system = """You are a senior clinical psychologist and cultural counseling supervisor.
Create a structured therapeutic response plan. Return ONLY valid JSON."""

    human = f"""CLIENT PROFILE:
- Primary emotion: {emotions.get('primary_emotion', 'neutral')} (intensity {emotions.get('intensity', 5)}/10)
- Secondary emotions: {emotions.get('secondary_emotions', [])}
- Therapeutic need: {emotions.get('therapeutic_need', 'validation')}
- Cultural region: {cultural.get('region', 'Universal')}
- Faith: {faith.get('faith', 'Not specified')}
- Openness to spiritual guidance: {faith.get('openness_to_spiritual_counselling', True)}
- Core values: {values.get('core_values', [])}
- Central conflict: {values.get('central_conflict', 'None')}
- Preferred modality: {prefs.get('modality', 'balanced')}
- {memory_str}

USER MESSAGE: "{user_message}"

RETRIEVED KNOWLEDGE CONTEXT:
{assembled_context[:800] if assembled_context else "None yet"}

Create a 4-component response plan:
{{
  "empathy_validation": "Specific empathy statement acknowledging their exact situation",
  "cultural_framing": "How to connect pain to a cultural/philosophical concept (or CBT if secular)",
  "psychological_integration": "Which evidence-based technique to apply and why",
  "micro_action": "One specific, small, actionable step the user can do TODAY",
  "therapeutic_goal": "validation|agency|meaning|safety|exploration",
  "selected_knowledge_domains": ["spiritual", "cbt", "cultural", "psychoedu", "local_resources"],
  "tone_instruction": "How the final response should sound (e.g., warm but grounded)"
}}"""

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        content = resp.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"): content = content[4:]
        return json.loads(content)
    except Exception as e:
        print(f"[ResponsePlanner] Error: {e}")
        primary_emotion = emotions.get("primary_emotion", "distress")
        return {
            "empathy_validation": f"I hear that you're experiencing {primary_emotion} right now, and your feelings are completely valid.",
            "cultural_framing": "The Bhagavad Gita teaches us about finding peace through acceptance and focused action.",
            "psychological_integration": "We'll use Cognitive Reframing to help shift perspective on this situation.",
            "micro_action": "Take 3 deep breaths and write down one thing you're grateful for today.",
            "therapeutic_goal": emotions.get("therapeutic_need", "validation"),
            "selected_knowledge_domains": ["spiritual", "cbt"],
            "tone_instruction": "Warm, empathetic, non-judgmental, and gently philosophical.",
        }
