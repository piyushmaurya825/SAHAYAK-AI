"""
Central configuration hub for SAHAYAK AI — Mental Health Companion.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Project Paths
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "knowledge_base" / "data"
DB_PATH = BASE_DIR / "data" / "mh_agent.db"
CHROMA_DIR = BASE_DIR / "data" / "chroma_store"
CRISIS_LOG = BASE_DIR / "data" / "crisis_log.jsonl"
CHAT_HISTORY_FILE = BASE_DIR / "data" / "chat_history.json"
MEMORY_FILE = BASE_DIR / "data" / "memory_store.json"

# Ensure data directories exist
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Ollama (Primary LLM)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gemma2:2b")
FAST_MODEL = os.getenv("FAST_MODEL", "gemma2:2b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# ChromaDB
CHROMA_COLLECTION_SPIRITUAL = "spiritual_knowledge"
CHROMA_COLLECTION_CBT = "cbt_techniques"
CHROMA_COLLECTION_CULTURAL = "cultural_practices"
CHROMA_COLLECTION_PSYCHOEDU = "psychoeducation"
CHROMA_COLLECTION_GUIDED = "guided_responses"
CHROMA_TOP_K = 4
CHROMA_RERANK_THRESHOLD = 0.50

# Safety Thresholds
CRISIS_SEVERITY_LEVELS = ["NONE", "LOW", "MODERATE", "ACUTE"]
SAFETY_RETRY_MAX = 1

# Emotion Detection (via Ollama LLM — direct single-pass)
EMOTION_TOP_K = 3                    # max emotions to retain
EMOTION_INTENSITY_THRESHOLD = 0.15  # minimum intensity to consider an emotion

# Langfuse Observability
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# Memory & History
MAX_SESSIONS_REMEMBERED = 10
MAX_CONVERSATION_TURNS = 20
CONTEXT_WINDOW_TOKENS = 6000

# Generation Parameters
GENERATION_TEMPERATURE = 0.7
GENERATION_MAX_TOKENS = 1200
GUARDRAIL_JUDGE_TEMPERATURE = 0.0

# Cultural & Regional Settings
SUPPORTED_REGIONS = [
    "South Asian", "East Asian", "Southeast Asian",
    "Middle Eastern", "African", "Western European",
    "North American", "Latin American", "Australian", "Universal"
]

SUPPORTED_FAITHS = [
    "Hindu", "Muslim", "Christian", "Buddhist", "Sikh",
    "Jain", "Jewish", "Atheist", "Agnostic", "Spiritual (non-religious)", "Other"
]

# Crisis Helplines (India-first, with international fallbacks)
CRISIS_HELPLINES = {
    "India": [
        {"name": "Tele-MANAS (Govt. of India)", "number": "1800-891-4416", "available": "24/7"},
        {"name": "KIRAN Mental Health Helpline", "number": "1800-599-0019", "available": "24/7"},
        {"name": "Vandrevala Foundation", "number": "+91-9999-666-555", "available": "24/7"},
        {"name": "NIMHANS Helpline", "number": "080-46110007", "available": "24/7"},
        {"name": "iCall (TISS)", "number": "9152987821", "available": "Mon-Sat 8am-10pm"},
    ],
    "International": [
        {"name": "IASP Crisis Centres", "number": "https://www.iasp.info/resources/Crisis_Centres/", "available": "24/7"},
    ],
}
DEFAULT_HELPLINE_REGION = "India"
