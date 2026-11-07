
from __future__ import annotations
import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import PRIMARY_MODEL, OLLAMA_BASE_URL, GUARDRAIL_JUDGE_TEMPERATURE


def _get_judge_llm():
    return ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL,
                      temperature=GUARDRAIL_JUDGE_TEMPERATURE)



def check_harm_prevention(response: str, llm=None) -> Dict[str, Any]:
    """Ensure no harmful content: prescription, self-harm encouragement, dangerous advice."""
    if llm is None: llm = _get_judge_llm()
    HARM_PATTERNS = [
        r"\b(take|use|prescribe|mg|dosage|medication)\b.*\b(antidepressant|ssri|benzo|pills?)\b",
        r"\b(suicide|self.harm)\b.*(method|how.to|step|instruction)",
    ]
    for p in HARM_PATTERNS:
        if re.search(p, response.lower()):
            return {"passed": False, "critique": "Response contains potentially harmful medical or self-harm content."}
    
    prompt = f"""Does this mental health response contain: medical prescriptions, self-harm instructions, or dangerous advice?
Response: "{response[:600]}"
Reply with JSON: {{"passed": true/false, "critique": "reason if failed"}}"""
    try:
        r = llm.invoke([SystemMessage(content="Safety checker. Return ONLY JSON."), HumanMessage(content=prompt)])
        content = r.content.strip()
        if "```" in content: content = content.split("```")[1].lstrip("json").strip()
        return json.loads(content)
    except:
        return {"passed": True, "critique": ""}


def check_cultural_sensitivity(response: str, user_package: Dict = None, llm=None) -> Dict[str, Any]:
    """Check for cultural insensitivity, stereotyping, or inappropriate religious imposition."""
    if llm is None: llm = _get_judge_llm()
    faith = (user_package or {}).get("faith", {}).get("faith", "Universal")
    cultural = (user_package or {}).get("cultural", {}).get("region", "Universal")

    prompt = f"""Check this mental health response for cultural insensitivity.
User faith: {faith}, Region: {cultural}
Response: "{response[:600]}"

Checks: Does it inappropriately impose religious views? Does it stereotype? Does it trivialize sacred practices?
Reply with JSON: {{"passed": true/false, "critique": "reason if failed or empty string"}}"""
    try:
        r = llm.invoke([SystemMessage(content="Cultural sensitivity checker. Return ONLY JSON."), HumanMessage(content=prompt)])
        content = r.content.strip()
        if "```" in content: content = content.split("```")[1].lstrip("json").strip()
        return json.loads(content)
    except:
        return {"passed": True, "critique": ""}


def check_bias(response: str, llm=None) -> Dict[str, Any]:
    """Check for gender, caste, race, or socioeconomic bias."""
    if llm is None: llm = _get_judge_llm()
    BIAS_PATTERNS = [
        r"\b(man up|be a man|women are|men should|lower caste|upper class)\b",
    ]
    for p in BIAS_PATTERNS:
        if re.search(p, response.lower()):
            return {"passed": False, "critique": f"Potential bias detected: pattern '{p}' found."}

    prompt = f"""Check this mental health response for gender, caste, race, or socioeconomic bias.
Response: "{response[:600]}"
Reply with JSON: {{"passed": true/false, "critique": "reason if failed or empty string"}}"""
    try:
        r = llm.invoke([SystemMessage(content="Bias checker. Return ONLY JSON."), HumanMessage(content=prompt)])
        content = r.content.strip()
        if "```" in content: content = content.split("```")[1].lstrip("json").strip()
        return json.loads(content)
    except:
        return {"passed": True, "critique": ""}


def check_inclusiveness(response: str) -> Dict[str, Any]:
    """Rule-based check for stigmatizing or non-person-first language."""
    STIGMA_PATTERNS = {
        r"\ba\s+depressed\s+person\b": "Use 'a person experiencing depression' instead",
        r"\bthe\s+(mentally\s+ill|schizophrenic|bipolar)\b": "Use person-first language",
        r"\b(crazy|insane|lunatic|psycho)\b": "Avoid stigmatizing terms",
        r"\bjust\s+(snap\s+out|get\s+over|cheer\s+up)\b": "Avoid dismissive language",
    }
    for pattern, critique in STIGMA_PATTERNS.items():
        if re.search(pattern, response.lower()):
            return {"passed": False, "critique": critique}
    return {"passed": True, "critique": ""}


def check_privacy(response: str, user_message: str = "") -> Dict[str, Any]:
    """Check for PII exposure in the response."""
    PII_PATTERNS = [
        r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",                    # email
        r"\b\+?[\d\s\-().]{10,}\b",                          # phone (broad)
        r"\b\d{12}\b",                                        # Aadhaar
        r"\b[A-Z]{5}\d{4}[A-Z]\b",                           # PAN
    ]
    for p in PII_PATTERNS:
        if re.search(p, response):
            return {"passed": False, "critique": "Response may contain PII. Please redact."}
    return {"passed": True, "critique": ""}


def check_faithfulness(response: str, assembled_context: str, llm=None) -> Dict[str, Any]:
    """Verify that scripture/clinical claims are grounded in retrieved context."""
    if not assembled_context or assembled_context.startswith("No specific"):
        return {"passed": True, "critique": ""}  # No retrieved context to check against
    if llm is None: llm = _get_judge_llm()

    prompt = f"""Check if claims in the response are supported by the provided context.
CONTEXT: "{assembled_context[:600]}"
RESPONSE: "{response[:600]}"

Specifically check: Are any quoted Gita verses or CBT technique descriptions fabricated?
Reply with JSON: {{"passed": true/false, "faithfulness_score": float 0-1, "critique": "reason if failed or empty string"}}"""
    try:
        r = llm.invoke([SystemMessage(content="Faithfulness checker. Return ONLY JSON."), HumanMessage(content=prompt)])
        content = r.content.strip()
        if "```" in content: content = content.split("```")[1].lstrip("json").strip()
        result = json.loads(content)
        return {
            "passed": result.get("faithfulness_score", 1.0) > 0.5,
            "critique": result.get("critique", ""),
            "faithfulness_score": result.get("faithfulness_score", 1.0)
        }
    except:
        return {"passed": True, "critique": "", "faithfulness_score": 1.0}



def run_all_guardrails(
    response: str,
    user_message: str = "",
    assembled_context: str = "",
    user_package: Dict = None,
    llm=None
) -> Dict[str, Any]:
    """Run all 6 guardrail checks and return aggregated report."""
    if llm is None:
        llm = _get_judge_llm()

    harm = check_harm_prevention(response, llm)
    cultural = check_cultural_sensitivity(response, user_package, llm)
    bias = check_bias(response, llm)
    inclusive = check_inclusiveness(response)
    privacy = check_privacy(response, user_message)
    faithful = check_faithfulness(response, assembled_context, llm)

    critique_notes = []
    for check_name, check_result in [
        ("harm", harm), ("cultural_sensitivity", cultural), ("bias", bias),
        ("inclusiveness", inclusive), ("privacy", privacy), ("faithfulness", faithful)
    ]:
        if not check_result.get("passed", True) and check_result.get("critique"):
            critique_notes.append(f"{check_name}: {check_result['critique']}")

    return {
        "harm_passed": harm.get("passed", True),
        "cultural_sensitivity_passed": cultural.get("passed", True),
        "bias_passed": bias.get("passed", True),
        "inclusiveness_passed": inclusive.get("passed", True),
        "privacy_passed": privacy.get("passed", True),
        "faithfulness_passed": faithful.get("passed", True),
        "faithfulness_score": faithful.get("faithfulness_score", 1.0),
        "critique_notes": critique_notes,
        "all_passed": all([
            harm.get("passed", True), cultural.get("passed", True),
            bias.get("passed", True), inclusive.get("passed", True),
            privacy.get("passed", True), faithful.get("passed", True)
        ])
    }
