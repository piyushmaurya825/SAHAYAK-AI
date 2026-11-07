"""
Main LLM response generator with therapeutic persona.
"""

from __future__ import annotations
import json
from typing import Any, Dict, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PRIMARY_MODEL, OLLAMA_BASE_URL, GENERATION_TEMPERATURE, GENERATION_MAX_TOKENS

THERAPEUTIC_SYSTEM_PROMPT = """You are a compassionate, culturally-competent mental health companion AI.

Your core principles:
1. EMPATHY FIRST: Acknowledge pain before offering perspective
2. CULTURAL RESPECT: Honor the user's cultural and spiritual values
3. NON-JUDGMENTAL: Never shame, lecture, or moralize
4. EVIDENCE-BASED: Ground all suggestions in retrieved clinical knowledge
5. SAFETY-FIRST: Never diagnose, prescribe, or replace professional care
6. PERSON-CENTERED: Follow the user's lead on depth of engagement

Communication style:
- Use accessible, warm language (not clinical jargon)
- Follow the structure of our clinical exemplars: validate pain -> offer grounding -> give one small step -> invite connection
- Blend philosophical wisdom naturally into the conversation
- Offer concrete, small steps, not overwhelming advice
- End with an open, inviting question that encourages reflection"""


def generate_response(
    user_message: str,
    user_package: Dict[str, Any],
    response_plan: Dict[str, Any],
    assembled_context: str,
    conversation_history: list = None,
    critique_notes: list = None,
    llm: Optional[ChatOllama] = None,
) -> str:
    """Generate the therapeutic response using the response plan and retrieved context."""
    if llm is None:
        llm = ChatOllama(
            model=PRIMARY_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=GENERATION_TEMPERATURE,
            num_predict=GENERATION_MAX_TOKENS,
        )

    emotions = user_package.get("emotions", {})
    prefs = user_package.get("preferences", {})
    cultural = user_package.get("cultural", {})
    faith = user_package.get("faith", {})

    # Build conversation history context
    history_str = ""
    if conversation_history:
        recent = conversation_history[-6:]  # last 3 exchanges
        history_str = "\n".join(
            f"{m.get('role','user').upper()}: {m.get('content','')[:300]}"
            for m in recent
        )

    # Critique/retry instructions
    critique_str = ""
    if critique_notes:
        critique_str = f"\n\nIMPORTANT - Your previous response had these issues to fix:\n{chr(10).join('- ' + n for n in critique_notes)}"

    human = f"""RESPONSE PLAN:
- Empathy statement: {response_plan.get('empathy_validation', '')}
- Cultural framing: {response_plan.get('cultural_framing', '')}
- Psychological technique: {response_plan.get('psychological_integration', '')}
- Micro-action: {response_plan.get('micro_action', '')}
- Tone: {response_plan.get('tone_instruction', 'warm and empathetic')}

USER CONTEXT:
- Primary emotion: {emotions.get('primary_emotion', 'distress')} (intensity: {emotions.get('intensity', 5)}/10)
- Cultural background: {cultural.get('region', 'Universal')}
- Faith/spirituality: {faith.get('faith', 'Not specified')} - Openness: {faith.get('openness_to_spiritual_counselling', True)}
- Preferred style: {prefs.get('language_style', 'formal English')}, modality: {prefs.get('modality', 'balanced')}

RETRIEVED KNOWLEDGE:
{assembled_context}

CONVERSATION HISTORY:
{history_str or "This is the first message."}

CURRENT USER MESSAGE: "{user_message}"
{critique_str}

Write a compassionate, culturally-appropriate therapeutic response following the plan above.
If a Therapeutic Exemplar is present in the retrieved knowledge, mirror its exact response structure:
1. Validation: Validate the emotion and difficulty immediately.
2. Grounding / Reframe: Offer a gentle perspective shift or philosophical grounding.
3. Micro-action: Give one small, realistic action for today.
4. Social Support & Question: Remind the user they don't have to carry it all alone and end with one reflective question.
Keep response focused, warm, and natural (120-250 words)."""

    try:
        resp = llm.invoke([SystemMessage(content=THERAPEUTIC_SYSTEM_PROMPT), HumanMessage(content=human)])
        return resp.content.strip()
    except Exception as e:
        print(f"[ResponseGenerator] Error: {e}")
        return (
            f"I hear you, and I want you to know that what you're feeling right now is valid and real. "
            f"You've taken a brave step just by sharing this.\n\n"
            f"The Bhagavad Gita reminds us in Chapter 2, Verse 47: *'You have a right to your actions, "
            f"but not to the fruits of your actions.'* This wisdom encourages us to focus on what we can "
            f"control — our present effort and care — rather than on outcomes we cannot.\n\n"
            f"One small step for today: take 5 minutes to write down what you're feeling without judgment. "
            f"Just let the words flow.\n\n"
            f"What feels like the most pressing thing on your mind right now?"
        )
