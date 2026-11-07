"""
Multi-tier safety risk assessment:
  Tier 1: Instant regex/keyword matching
  Tier 2: Ollama LLM as C-SSRS classifier
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    PRIMARY_MODEL, OLLAMA_BASE_URL, GUARDRAIL_JUDGE_TEMPERATURE,
    CRISIS_HELPLINES, DEFAULT_HELPLINE_REGION
)


# Tier 1: Keyword patterns
CRISIS_PATTERNS = {
    "ACUTE": [
        r"\bsuicid\w*\b", r"\bkill\s+myself\b", r"\bend\s+(my\s+)?life\b",
        r"\bwant\s+to\s+die\b", r"\bno\s+reason\s+to\s+live\b",
        r"\bbetter\s+off\s+dead\b", r"\bending\s+it\s+all\b",
        r"\bself[- ]harm\b", r"\bcut\s+myself\b", r"\bhurt\s+myself\b",
    ],
    "MODERATE": [
        r"\bdepressed\b", r"\bhopeless\b", r"\bworthless\b",
        r"\bno\s+hope\b", r"\bgive\s+up\b", r"\bcan.t\s+(go\s+on|continue)\b",
        r"\bextremely\s+sad\b", r"\bbreaking\s+down\b",
    ],
}


def tier1_keyword_check(text: str) -> Tuple[str, List[str]]:
    """Fast regex scan. Returns (severity, matched_patterns)."""
    text_lower = text.lower()
    for severity in ["ACUTE", "MODERATE"]:
        matches = [p for p in CRISIS_PATTERNS[severity] if re.search(p, text_lower)]
        if matches:
            return severity, matches
    return "NONE", []


# Tier 2: Ollama LLM C-SSRS Classifier (deep local analysis)
def tier2_llm_cssrs(text: str) -> Dict[str, Any]:
    """Use Ollama LLM with C-SSRS protocol to classify risk level."""
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL,
                         temperature=GUARDRAIL_JUDGE_TEMPERATURE)
        system = """You are a clinical risk assessment AI trained on the Columbia Suicide Severity Rating Scale (C-SSRS).
Analyze text for suicide/self-harm risk. Return ONLY valid JSON. Be conservative - if in doubt, rate higher."""

        human = f"""User message: "{text}"

Assess risk using C-SSRS criteria:
- Level 0: No ideation
- Level 1: Passive (wish to be dead, not living)
- Level 2: Active (thoughts of killing self, no plan)
- Level 3: Active with method (has thought about how)
- Level 4: Active with intent (plans to act)
- Level 5: Active with plan and intent (imminent)

Return JSON:
{{
  "c_ssrs_level": integer 0-5,
  "severity": "NONE|LOW|MODERATE|ACUTE",
  "is_crisis": boolean,
  "triggers": ["list of identified risk triggers"],
  "protective_factors": ["list of identified protective factors"],
  "reasoning": "brief clinical reasoning"
}}"""

        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        content = resp.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"): content = content[4:]
        return json.loads(content)
    except Exception as e:
        print(f"[SafetyAssessor] LLM C-SSRS error: {e}")
        return {
            "c_ssrs_level": 0, "severity": "NONE", "is_crisis": False,
            "triggers": [], "protective_factors": [], "reasoning": "Assessment unavailable"
        }


def assess_safety(text: str) -> Dict[str, Any]:
    """
    2-tier local safety assessment pipeline.

    Tier 1: Instant regex/keyword matching
      - ACUTE match → immediate crisis response (no LLM needed)
      - MODERATE match → escalate to Tier 2 for deeper analysis

    Tier 2: Ollama LLM as C-SSRS classifier
      - Runs on any MODERATE or ambiguous/distress signal
      - Conservative: when in doubt, rates higher
    """
    # Tier 1: Fast keyword check
    t1_severity, t1_matches = tier1_keyword_check(text)

    # If ACUTE detected at Tier 1, short-circuit immediately
    if t1_severity == "ACUTE":
        return {
            "is_crisis": True,
            "severity": "ACUTE",
            "c_ssrs_level": 4,
            "triggers": t1_matches,
            "moderation_flags": {},
            "assessment_tier": 1,
        }

    # Tier 2: Local LLM C-SSRS deep analysis for MODERATE or any distress signals
    if t1_severity == "MODERATE":
        t2_result = tier2_llm_cssrs(text)
        return {
            "is_crisis": t2_result.get("is_crisis", False),
            "severity": t2_result.get("severity", "LOW"),
            "c_ssrs_level": t2_result.get("c_ssrs_level", 0),
            "triggers": t2_result.get("triggers", []) + t1_matches,
            "protective_factors": t2_result.get("protective_factors", []),
            "moderation_flags": {},
            "assessment_tier": 2,
        }

    # No risk signals detected
    return {
        "is_crisis": False,
        "severity": "NONE",
        "c_ssrs_level": 0,
        "triggers": [],
        "moderation_flags": {},
        "assessment_tier": 1,
    }


def get_helplines(region: str = DEFAULT_HELPLINE_REGION) -> List[Dict]:
    """Return crisis helplines for the given region."""
    helplines = CRISIS_HELPLINES.get(region, CRISIS_HELPLINES[DEFAULT_HELPLINE_REGION])
    if region not in CRISIS_HELPLINES:
        helplines = CRISIS_HELPLINES[DEFAULT_HELPLINE_REGION]
    return helplines
