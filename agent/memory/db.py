"""
SQLAlchemy models and SQLite database setup for persistent memory storage.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, Boolean,
    create_engine, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DB_PATH

Base = declarative_base()
_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{DB_PATH}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        # Enable WAL mode for better concurrency
        @event.listens_for(_engine, "connect")
        def set_wal(conn, _):
            conn.execute("PRAGMA journal_mode=WAL")
        Base.metadata.create_all(_engine)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal()



class UserProfile(Base):
    
    __tablename__ = "user_profiles"

    user_id = Column(String, primary_key=True, index=True)
    region = Column(String, default="Universal")
    community = Column(String, default="Unknown")
    faith = Column(String, default="Spiritual (non-religious)")
    spirituality_level = Column(String, default="moderate")
    openness_to_spiritual = Column(Boolean, default=True)
    core_values_json = Column(Text, default="[]")          # JSON list
    preferred_tone = Column(String, default="empathetic")
    preferred_modality = Column(String, default="balanced")
    language_style = Column(String, default="formal English")
    collectivism_index = Column(Float, default=0.5)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def core_values(self) -> list:
        return json.loads(self.core_values_json or "[]")

    @core_values.setter
    def core_values(self, val: list):
        self.core_values_json = json.dumps(val)


class ConversationSession(Base):
    
    __tablename__ = "conversation_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    session_id = Column(String, index=True, unique=True)
    date = Column(String)               # YYYY-MM-DD
    time = Column(String)               # HH:MM
    root_cause_redacted = Column(Text)
    primary_emotion = Column(String)
    spiritual_verse_shown = Column(String)
    cbt_technique_shown = Column(String)
    response_kind = Column(String, default="normal")
    is_crisis = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
  
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    session_id = Column(String, index=True)
    role = Column(String)               # "user" | "assistant"
    content = Column(Text)
    kind = Column(String, default="normal")
    timestamp = Column(DateTime, default=datetime.utcnow)


class SessionMetricsRecord(Base):
   
    __tablename__ = "session_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True, unique=True)
    user_id = Column(String, index=True)
    cultural_relevance = Column(Float, default=0.0)
    empathy_score = Column(Float, default=0.0)
    safety_score = Column(Float, default=1.0)
    grounding_score = Column(Float, default=0.0)
    user_feedback = Column(Integer, nullable=True)
    audit_trail = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


get_engine()
