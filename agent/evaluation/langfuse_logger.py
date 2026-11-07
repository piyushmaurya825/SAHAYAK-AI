"""
Langfuse observability integration for LLM trace logging.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import LANGFUSE_ENABLED, LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

_langfuse_client = None


def get_langfuse():
    global _langfuse_client
    if not LANGFUSE_ENABLED:
        return None
    if _langfuse_client is None:
        try:
            from langfuse import Langfuse
            _langfuse_client = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST,
            )
            print("[Langfuse] Connected successfully.")
        except Exception as e:
            print(f"[Langfuse] Connection failed: {e}")
            _langfuse_client = None
    return _langfuse_client


def log_session_trace(
    session_id: str,
    user_id: str,
    user_message: str,
    response: str,
    metrics: Dict[str, Any],
    metadata: Dict[str, Any] = None,
) -> Optional[str]:
    """Log a complete session trace to Langfuse."""
    lf = get_langfuse()
    if lf is None:
        return None

    try:
        trace = lf.trace(
            name="mental_health_session",
            id=session_id,
            user_id=user_id,
            input={"user_message": user_message[:500]},
            output={"response": response[:500]},
            metadata={
                "empathy_score": metrics.get("empathy_score"),
                "safety_score": metrics.get("safety_score"),
                "cultural_relevance": metrics.get("cultural_relevance_score"),
                "grounding": metrics.get("grounding_score"),
                **(metadata or {}),
            }
        )
        lf.flush()
        return trace.id
    except Exception as e:
        print(f"[Langfuse] Trace logging failed: {e}")
        return None


def log_guardrail_event(session_id: str, guardrail_report: Dict, retry_count: int = 0) -> None:
    """Log guardrail check results as a span."""
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.event(
            name="guardrail_check",
            trace_id=session_id,
            input={"retry_count": retry_count},
            output=guardrail_report,
            metadata={"all_passed": guardrail_report.get("all_passed", True)}
        )
    except Exception as e:
        print(f"[Langfuse] Event logging failed: {e}")
