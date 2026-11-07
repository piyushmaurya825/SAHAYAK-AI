"""
Assembles retrieved chunks from all domains into a structured context string.
"""

from __future__ import annotations
from typing import Any, Dict, List

from config import CHROMA_RERANK_THRESHOLD


def _format_chunk(chunk: Dict[str, Any], domain_label: str) -> str:
    meta = chunk.get("metadata", {})
    text = chunk.get("text", "")
    relevance = chunk.get("relevance", 0)
    
    if relevance < CHROMA_RERANK_THRESHOLD:
        return ""
    
    if domain_label == "GUIDED_RESPONSE" or "EXEMPLAR" in domain_label:
        strategy = meta.get("support_strategy", "")
        guided = meta.get("guided_response") or text
        cat = meta.get("category", "")
        return (f"[Therapeutic Exemplar - Topic: {cat}]\n"
                f"Support Strategy: {strategy}\n"
                f"Target Response Style: {guided}")
    elif domain_label == "SPIRITUAL":
        return (f"[Bhagavad Gita {meta.get('id','')}: {meta.get('concept','').upper()}]\n"
                f"{text}\n"
                f"→ Therapeutic mapping: {meta.get('psychology','')}")
    elif domain_label == "CBT":
        return (f"[CBT Technique: {meta.get('name','')}]\n"
                f"{text}")
    elif domain_label == "CULTURAL":
        return (f"[Cultural Practice ({meta.get('culture','')}): {meta.get('practice','')}]\n"
                f"{text}")
    elif domain_label == "PSYCHOEDU":
        return (f"[Psychoeducation: {meta.get('title','')}]\n"
                f"{text[:300]}...")
    else:
        return f"[Resource]\n{text[:300]}"


def assemble_context(
    retrieved: Dict[str, List[Dict]],
    local_resources: List[Dict] = None,
    max_chars: int = 3000,
) -> str:
    """Combine all retrieved chunks into a structured prompt context."""
    sections = []
    total_chars = 0

    domain_labels = {
        "guided_responses": "GUIDED_RESPONSE",
        "spiritual": "SPIRITUAL",
        "cbt": "CBT",
        "cultural": "CULTURAL",
        "psychoedu": "PSYCHOEDU",
    }

    for domain, label in domain_labels.items():
        chunks = retrieved.get(domain, [])
        if not chunks:
            continue
        domain_parts = []
        for chunk in chunks[:2]:  # max 2 per domain
            formatted = _format_chunk(chunk, label)
            if formatted and (total_chars + len(formatted)) < max_chars:
                domain_parts.append(formatted)
                total_chars += len(formatted)
        if domain_parts:
            sections.append(f"\n### {label}\n" + "\n\n".join(domain_parts))

    if local_resources:
        res_lines = []
        for r in local_resources[:3]:
            res_lines.append(f"• {r.get('name','')}: {r.get('number','')} ({r.get('available','')})")
        sections.append("\n### LOCAL SUPPORT RESOURCES\n" + "\n".join(res_lines))

    if not sections:
        return "No specific knowledge retrieved. Respond with general empathy and evidence-based support."

    return "\n".join(sections)
