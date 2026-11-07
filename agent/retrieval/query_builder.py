"""
Builds specialized sub-queries for each knowledge domain using HyDE-inspired approach.
"""

from __future__ import annotations
import json
from typing import Any, Dict, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FAST_MODEL, OLLAMA_BASE_URL

def build_domain_queries(
    user_message: str,
    user_package: Dict[str, Any],
    llm: Optional[ChatOllama] = None
) -> Dict[str, str]:
    """Generate specialized retrieval queries for each knowledge domain."""
    if llm is None:
        llm = ChatOllama(model=FAST_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)

    user_package = user_package or {}
    emotions = user_package.get("emotions", {}) or {}
    cultural = user_package.get("cultural", {}) or {}
    values = user_package.get("values", {}) or {}

    system = "You are a retrieval query specialist for a mental health RAG system. Return ONLY valid JSON."
    human = f"""User message: "{user_message}"
Primary emotion: {emotions.get('primary_emotion', 'distress')}
Cultural region: {cultural.get('region', 'Universal')}
Core values: {values.get('core_values', [])}
Central conflict: {values.get('central_conflict', '')}

Generate specialized search queries for each domain. Make queries dense with relevant terms.
Return JSON:
{{
  "guided": "query for matching therapeutic strategy and exemplar response",
  "spiritual": "query for Bhagavad Gita / philosophical wisdom relevant to situation",
  "cbt": "query for CBT technique matching the cognitive pattern",
  "cultural": "query for cultural coping practice matching user background",
  "psychoedu": "query for psychoeducation about the condition/emotion",
  "general": "general semantic query capturing core issue"
}}"""

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        content = resp.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"): content = content[4:]
        return json.loads(content)
    except Exception as e:
        print(f"[QueryBuilder] Error: {e}")
        # Fallback: use user message directly
        return {
            "guided": f"{user_message} therapeutic strategy support",
            "spiritual": f"{user_message} detachment acceptance equanimity soul karma",
            "cbt": f"{user_message} cognitive reframing thoughts feelings behavior",
            "cultural": f"{user_message} {cultural.get('region', 'universal')} coping practice",
            "psychoedu": f"{user_message} mental health understanding symptoms",
            "general": user_message,
        }
