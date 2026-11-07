"""
Read/write user profiles and conversation sessions from SQLite.
"""

from __future__ import annotations
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.memory.db import get_session, UserProfile, ConversationSession, ChatMessage, SessionMetricsRecord


def _redact(text: str, limit: int = 200) -> str:
    """Remove PII before persistence."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", text)
    text = re.sub(r"\+?\d[\d\s().-]{8,}\d", "[phone]", text)
    return text.strip()[:limit]


class ProfileStore:

    def get(self, user_id: str) -> Optional[Dict]:
        with get_session() as session:
            p = session.get(UserProfile, user_id)
            if p is None:
                return None
            return {
                "user_id": p.user_id,
                "region": p.region,
                "community": p.community,
                "faith": p.faith,
                "spirituality_level": p.spirituality_level,
                "openness_to_spiritual": p.openness_to_spiritual,
                "core_values": p.core_values,
                "preferred_tone": p.preferred_tone,
                "preferred_modality": p.preferred_modality,
                "language_style": p.language_style,
                "collectivism_index": p.collectivism_index,
            }

    def upsert(self, user_id: str, updates: Dict[str, Any]) -> None:
        with get_session() as session:
            p = session.get(UserProfile, user_id)
            if p is None:
                p = UserProfile(user_id=user_id)
                session.add(p)
            for k, v in updates.items():
                if k == "core_values" and isinstance(v, list):
                    p.core_values = v
                elif hasattr(p, k):
                    setattr(p, k, v)
            session.commit()


class ConversationStore:

    def get_recent_sessions(self, user_id: str, limit: int = 5) -> List[Dict]:
        with get_session() as session:
            rows = (
                session.query(ConversationSession)
                .filter_by(user_id=user_id)
                .order_by(ConversationSession.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "session_id": r.session_id,
                    "date": r.date,
                    "primary_emotion": r.primary_emotion,
                    "root_cause": r.root_cause_redacted,
                    "verse": r.spiritual_verse_shown,
                    "cbt": r.cbt_technique_shown,
                    "is_crisis": r.is_crisis,
                }
                for r in rows
            ]

    def get_past_verses(self, user_id: str) -> List[str]:
        with get_session() as session:
            rows = (
                session.query(ConversationSession.spiritual_verse_shown)
                .filter_by(user_id=user_id)
                .filter(ConversationSession.spiritual_verse_shown.isnot(None))
                .all()
            )
            return [r[0] for r in rows if r[0]]

    def save_session(
        self,
        user_id: str,
        session_id: str,
        root_cause: str = "",
        primary_emotion: str = "neutral",
        verse: str = "",
        cbt: str = "",
        is_crisis: bool = False,
        response_kind: str = "normal",
    ) -> None:
        with get_session() as session:
            now = datetime.utcnow()
            record = ConversationSession(
                user_id=user_id,
                session_id=session_id,
                date=now.strftime("%Y-%m-%d"),
                time=now.strftime("%H:%M"),
                root_cause_redacted=_redact(root_cause),
                primary_emotion=primary_emotion,
                spiritual_verse_shown=verse,
                cbt_technique_shown=cbt,
                is_crisis=is_crisis,
                response_kind=response_kind,
            )
            session.merge(record)
            session.commit()

    def save_messages(self, user_id: str, session_id: str, messages: List[Dict]) -> None:
        with get_session() as session:
            for m in messages:
                msg = ChatMessage(
                    user_id=user_id,
                    session_id=session_id,
                    role=m.get("role", "user"),
                    content=_redact(m.get("content", ""), limit=2000),
                    kind=m.get("kind", "normal"),
                )
                session.add(msg)
            session.commit()

    def get_messages(self, user_id: str, session_id: str) -> List[Dict]:
        with get_session() as session:
            rows = (
                session.query(ChatMessage)
                .filter_by(user_id=user_id, session_id=session_id)
                .order_by(ChatMessage.timestamp.asc())
                .all()
            )
            return [{"role": r.role, "content": r.content, "kind": r.kind} for r in rows]


class MetricsStore:
    
    def save(self, session_id: str, user_id: str, metrics: Dict[str, Any]) -> None:
        with get_session() as session:
            record = SessionMetricsRecord(
                session_id=session_id,
                user_id=user_id,
                cultural_relevance=metrics.get("cultural_relevance_score", 0.0),
                empathy_score=metrics.get("empathy_score", 0.0),
                safety_score=metrics.get("safety_score", 1.0),
                grounding_score=metrics.get("grounding_score", 0.0),
                user_feedback=metrics.get("user_feedback"),
                audit_trail=metrics.get("audit_trail", ""),
            )
            session.merge(record)
            session.commit()

    def get_average_scores(self, user_id: str) -> Dict[str, float]:
        with get_session() as session:
            from sqlalchemy import func
            result = (
                session.query(
                    func.avg(SessionMetricsRecord.empathy_score),
                    func.avg(SessionMetricsRecord.safety_score),
                    func.avg(SessionMetricsRecord.cultural_relevance),
                )
                .filter_by(user_id=user_id)
                .first()
            )
            return {
                "avg_empathy": round(result[0] or 0.0, 2),
                "avg_safety": round(result[1] or 0.0, 2),
                "avg_cultural_relevance": round(result[2] or 0.0, 2),
            }


profile_store = ProfileStore()
conversation_store = ConversationStore()
metrics_store = MetricsStore()
