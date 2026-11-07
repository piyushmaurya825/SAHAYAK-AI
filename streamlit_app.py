
from __future__ import annotations

import json
import re
import time
import uuid
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from config import OLLAMA_BASE_URL
from agent.graph import build_mental_health_agent, make_initial_state
from agent.memory.profile_store import conversation_store, metrics_store


LOGO_PATH = Path(__file__).parent / "assets" / "sahayak_logo.jpg"

st.set_page_config(
    page_title="SAHAYAK AI",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

WELCOME_MSG = (
    "Hello, How are you feeling today?"
)
ASSISTANT_AVATAR = "🌿"
USER_AVATAR = "🙂"
CHAT_HISTORY_FILE = Path("data") / "chat_history.json"
CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

ss = st.session_state


class ChatStore:
    def __init__(self, path: Path = CHAT_HISTORY_FILE):
        self.path = path

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def list(self, user_id: str) -> list:
        chats = self._read().get(user_id, [])
        return sorted(chats, key=lambda c: (c.get("date", ""), c.get("time", "")), reverse=True)

    def get(self, user_id: str, chat_id: str) -> Optional[dict]:
        return next((c for c in self._read().get(user_id, []) if c["id"] == chat_id), None)

    def upsert(self, user_id: str, chat_id: str, messages: list) -> None:
        data = self._read()
        chats = data.setdefault(user_id, [])
        chat = next((c for c in chats if c["id"] == chat_id), None)
        if chat is None:
            first_user = next((m["content"] for m in messages if m["role"] == "user"), "New chat")
            title = first_user.strip().replace("\n", " ")
            chat = {
                "id": chat_id,
                "date": time.strftime("%Y-%m-%d"),
                "time": time.strftime("%H:%M"),
                "title": (title[:40] + "…") if len(title) > 40 else title,
            }
            chats.append(chat)
        chat["messages"] = [
            {"role": m["role"], "content": m["content"], "kind": m.get("kind", "normal")}
            for m in messages
        ]
        self._write(data)

    def delete(self, user_id: str, chat_id: str) -> None:
        data = self._read()
        data[user_id] = [c for c in data.get(user_id, []) if c["id"] != chat_id]
        self._write(data)


CHATS = ChatStore()


def current_user_id() -> str:
    return (ss.get("user_id") or "").strip() or "usr_001"


def reset_conversation() -> None:
    ss.messages = [{"role": "assistant", "content": WELCOME_MSG, "kind": "normal"}]
    ss.chat_id = None
    ss.session_id = None
    ss.conversation_history = []


def toggle_theme() -> None:
    ss.dark_mode = not ss.get("dark_mode", False)


def open_chat(chat_id: str) -> None:
    chat = CHATS.get(current_user_id(), chat_id)
    if not chat:
        return
    ss.messages = chat.get("messages", [])
    ss.chat_id = chat_id
    ss.conversation_history = [
        {"role": m["role"], "content": m["content"]}
        for m in ss.messages
        if m["role"] in ("user", "assistant")
    ]
    ss.session_id = None  # next message starts fresh session


def delete_chat(chat_id: str) -> None:
    CHATS.delete(current_user_id(), chat_id)
    if ss.get("chat_id") == chat_id:
        reset_conversation()
    st.toast("Chat deleted.", icon="🗑️")


def submit_feedback(score: int, session_id: str) -> None:
    """Save thumbs-up/down or star rating to SQLite metrics."""
    if session_id:
        try:
            metrics_store.save(session_id, current_user_id(), {"user_feedback": score})
            st.toast("Thanks for your feedback! 💙", icon="✅")
        except Exception:
            pass



if "agent_app" not in ss:
    with st.spinner("🌿 Initializing SAHAYAK AI…"):
        ss.agent_app = build_mental_health_agent()
        ss.agent_initialized = True

if "dark_mode" not in ss:
    ss.dark_mode = False
if "messages" not in ss:
    reset_conversation()
if "conversation_history" not in ss:
    ss.conversation_history = []
if "session_id" not in ss:
    ss.session_id = None
if "last_session_id" not in ss:
    ss.last_session_id = None


LIGHT = dict(
    bg="#f8f9fd", side="#eef0f7", text="#2d3748", muted="#718096",
    card="#ffffff", input="#edf2ff", border="#dde3f0", btn="#ffffff",
    accent="#6c63ff", accent2="#f6ad55", crisis="#fff5f5", crisis_border="#fc8181",
)
DARK = dict(
    bg="#0d1117", side="#161b22", text="#e6edf3", muted="#8b949e",
    card="#1c2128", input="#1c2128", border="#30363d", btn="#21262d",
    accent="#7c6fff", accent2="#fbbf24", crisis="#2d1515", crisis_border="#f87171",
)
c = DARK if ss.dark_mode else LIGHT

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, .stApp {{
        font-family: 'Inter', sans-serif;
        background: {c['bg']};
        color: {c['text']};
    }}

    /* Header */
    header[data-testid="stHeader"] {{
        background: {c['bg']};
        border-bottom: 1px solid {c['border']};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] > div {{
        background: {c['side']};
        border-right: 1px solid {c['border']};
    }}

    /* All text */
    .stApp, .stApp p, .stApp li, .stApp label, .stApp h1,
    .stApp h2, .stApp h3, .stApp h4, .stApp span,
    .stApp div[data-testid="stMarkdownContainer"] {{
        color: {c['text']};
    }}
    .stApp [data-testid="stCaptionContainer"], .stApp small {{
        color: {c['muted']};
    }}

    /* Chat bubbles */
    [data-testid="stChatMessage"] {{
        background: {c['card']};
        border-radius: 16px;
        border: 1px solid {c['border']};
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s ease;
    }}
    [data-testid="stChatMessage"]:hover {{
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }}

    /* Crisis message special styling */
    .crisis-box {{
        background: {c['crisis']};
        border: 2px solid {c['crisis_border']};
        border-radius: 16px;
        padding: 1.25rem;
    }}

    /* Input */
    .stApp input, .stApp textarea {{
        background: {c['input']} !important;
        color: {c['text']} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stChatInput"], [data-testid="stChatInput"] > div {{
        background: {c['input']};
        border-color: {c['border']};
        border-radius: 16px !important;
    }}
    [data-testid="stBottom"], [data-testid="stBottom"] > div {{
        background: {c['bg']};
    }}
    [data-testid="stBottom"] > div {{ padding-bottom: 2rem; }}

    /* Buttons — Universal Base */
    .stApp button,
    .stApp .stButton > button,
    .stApp [data-testid="baseButton-secondary"],
    .stApp [data-testid="stBaseButton-secondary"],
    section[data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] [data-testid="stColumn"] button,
    section[data-testid="stSidebar"] .stButton > button {{
        background: {c['card']} !important;
        background-color: {c['card']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
        transition: all 0.2s ease !important;
    }}

    /* Ensure text inside buttons is always crisp and visible */
    .stApp button p,
    .stApp button span,
    .stApp button div,
    section[data-testid="stSidebar"] button p,
    section[data-testid="stSidebar"] button span,
    section[data-testid="stSidebar"] [data-testid="stColumn"] button p,
    section[data-testid="stSidebar"] [data-testid="stColumn"] button span {{
        color: {c['text']} !important;
        font-weight: 500 !important;
    }}

    /* Button hover state */
    .stApp button:hover,
    .stApp .stButton > button:hover,
    section[data-testid="stSidebar"] button:hover,
    section[data-testid="stSidebar"] [data-testid="stColumn"] button:hover {{
        background: {c['input']} !important;
        background-color: {c['input']} !important;
        border-color: {c['accent']} !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(108, 99, 255, 0.18) !important;
    }}
    .stApp button:hover p,
    .stApp button:hover span,
    section[data-testid="stSidebar"] button:hover p,
    section[data-testid="stSidebar"] button:hover span {{
        color: {c['accent']} !important;
    }}

    /* Left-align chat history labels with clean ellipsis */
    section[data-testid="stSidebar"] [data-testid="stColumn"]:first-child button {{
        text-align: left !important;
        justify-content: flex-start !important;
        padding-left: 0.75rem !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stColumn"]:first-child button p {{
        text-align: left !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}

    /* Prominent CTA for '+ New Conversation' */
    .st-key-new_conv_btn button {{
        background: linear-gradient(135deg, {c['accent']}, #5548e0) !important;
        background-color: {c['accent']} !important;
        border: none !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(108, 99, 255, 0.3) !important;
    }}
    .st-key-new_conv_btn button p,
    .st-key-new_conv_btn button span {{
        color: #ffffff !important;
        font-weight: 600 !important;
    }}
    .st-key-new_conv_btn button:hover {{
        background: linear-gradient(135deg, #5b52e0, #473bc4) !important;
        background-color: #5b52e0 !important;
        box-shadow: 0 6px 16px rgba(108, 99, 255, 0.45) !important;
    }}
    .st-key-new_conv_btn button:hover p,
    .st-key-new_conv_btn button:hover span {{
        color: #ffffff !important;
    }}

    /* Sidebar logo styling */
    section[data-testid="stSidebar"] [data-testid="stImage"] img {{
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 0.25rem;
    }}

    /* Dividers */
    .stApp hr {{ border-color: {c['border']}; }}

    /* Main column constraints — generous top padding to clear Streamlit header bar */
    .block-container {{
        padding-top: 4.5rem !important;
        max-width: 820px;
    }}

    /* Hide Streamlit chrome */
    #MainMenu, footer {{ visibility: hidden; }}

    /* App title — padded top and bottom with generous line-height so text is never cut off */
    .app-title {{
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 0.35rem;
        padding-top: 0.6rem;
        padding-bottom: 0.3rem;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.35;
        letter-spacing: -0.01em;
        background: linear-gradient(135deg, {c['accent']}, {c['accent2']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
    }}
    .app-subtitle {{
        text-align: center;
        color: {c['muted']};
        font-size: 0.88rem;
        margin-bottom: 1.5rem;
    }}

    /* Theme toggle button — top-right */
    .st-key-theme_btn {{
        position: fixed;
        top: 0.55rem;
        right: 7.5rem;
        z-index: 1000000;
        width: auto;
    }}
    .st-key-theme_btn button {{
        border-radius: 999px !important;
        padding: 0.15rem 0.7rem !important;
        font-size: 1rem !important;
    }}

    /* Footer disclaimer */
    .footer-note {{
        position: fixed;
        left: 0; right: 0; bottom: 5px;
        text-align: center;
        font-size: 0.65rem;
        line-height: 1.3;
        color: {c['muted']};
        z-index: 999;
        padding: 0 1rem;
    }}

    /* Metrics / feedback row */
    .metric-pill {{
        display: inline-block;
        background: {c['card']};
        border: 1px solid {c['border']};
        border-radius: 999px;
        padding: 0.15rem 0.6rem;
        font-size: 0.72rem;
        color: {c['muted']};
        margin: 0 0.15rem;
    }}

    /* Sidebar chat list */
    .chat-item-date {{
        font-size: 0.75rem;
        font-weight: 600;
        color: {c['muted']};
        margin: 0.5rem 0 0.2rem 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Status indicator */
    .status-dot {{
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }}
    .status-online {{ background: #48bb78; }}
    .status-offline {{ background: #fc8181; }}
    </style>

    <h1 class="app-title">SAHAYAK AI</h1>
    <p class="app-subtitle">
        Your Companion for Mental Wellness · Listen · Support · Empower
    </p>
    <div class="footer-note">
        I am an AI companion, not a therapist. I cannot provide medical advice or emergency services.
        If you are in immediate danger, please contact emergency services or a crisis helpline.
    </div>
    """,
    unsafe_allow_html=True,
)

# Theme toggle button
st.button(
    "☀️" if ss.dark_mode else "🌙",
    key="theme_btn",
    on_click=toggle_theme,
    help="Switch to light mode" if ss.dark_mode else "Switch to dark mode",
)


def typing_effect(text: str):
    for token in re.split(r"(\s+)", text):
        yield token
        time.sleep(0.010)


def render_message(m: dict) -> None:
    avatar = ASSISTANT_AVATAR if m["role"] == "assistant" else USER_AVATAR
    with st.chat_message(m["role"], avatar=avatar):
        if m.get("kind") == "crisis":
            st.error(m["content"], icon="🚨")
        else:
            st.markdown(m["content"])


def run_agent_turn(user_message: str) -> tuple[str, str]:
    """
    Returns (reply_text, response_kind).
    """
    user_id = current_user_id()
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    thread_id = f"thread_{user_id}_{session_id}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = make_initial_state(
        user_id=user_id,
        message=user_message,
        conversation_history=list(ss.conversation_history),
        session_id=session_id,
    )

    reply, kind, new_session_id = "", "normal", session_id

    try:
        for event in ss.agent_app.stream(initial_state, config):
            if isinstance(event, dict):
                for node_name, node_state in event.items():
                    if not isinstance(node_state, dict):
                        continue
                    # Capture final response from either path
                    if node_name in ("finalize_response", "crisis_escalation"):
                        if node_state.get("final_response"):
                            reply = node_state["final_response"]
                            kind = node_state.get("response_kind", "normal")
                        if node_state.get("session_id"):
                            new_session_id = node_state["session_id"]
    except Exception as e:
        print(f"[AgentTurn] Pipeline error: {e}")
        reply = (
            "I'm here with you. I encountered a brief technical difficulty, but I want you to know "
            "your feelings are valid.\n\n"
            "Could you share a little more about what's on your mind? I'm listening."
        )
        kind = "normal"

    # Fallback if no response captured
    if not reply:
        reply = (
            "Thank you for sharing that with me. I'm here to listen.\n\n"
            "Could you tell me more about what you're going through?"
        )

    ss.session_id = new_session_id
    ss.last_session_id = new_session_id
    return reply, kind


for m in ss.messages:
    render_message(m)


if prompt := st.chat_input("Share what's on your mind…", key="chat_box"):
    prompt = prompt.strip()
    if not prompt:
        st.stop()

    # Show user message immediately
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    ss.messages.append({"role": "user", "content": prompt, "kind": "normal"})

    # Assign chat ID on first message
    if not ss.chat_id:
        ss.chat_id = uuid.uuid4().hex

    # Run agent
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("🌸 Reflecting on your message…"):
            reply, kind = run_agent_turn(prompt)

        # Stream or display reply
        if kind == "crisis":
            st.error(reply, icon="🚨")
        else:
            if hasattr(st, "write_stream"):
                st.write_stream(typing_effect(reply))
            else:
                st.markdown(reply)

        # Feedback widget for non-crisis responses
        if kind != "crisis" and ss.last_session_id:
            with st.expander("💬 Was this helpful?", expanded=False):
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button("👍", key=f"fb_up_{ss.last_session_id}", help="Helpful"):
                        submit_feedback(5, ss.last_session_id)
                with col2:
                    if st.button("👎", key=f"fb_dn_{ss.last_session_id}", help="Not helpful"):
                        submit_feedback(1, ss.last_session_id)

    # Append assistant message
    ss.messages.append({"role": "assistant", "content": reply, "kind": kind})

    # Update conversation history for multi-turn context
    ss.conversation_history.extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply},
    ])
    ss.conversation_history = ss.conversation_history[-20:]

    CHATS.upsert(current_user_id(), ss.chat_id, ss.messages)
    st.rerun()


started = len(ss.messages) > 1

with st.sidebar:
    # Header — Logo + Brand name
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    st.markdown(
        f"""
        <div style="text-align:center; padding: 0.2rem 0 1rem 0;">
            <div style="font-weight:700; font-size:1.2rem; color:{c['text']};"></div>
            <div style="font-size:0.75rem; color:{c['muted']};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # User ID
    st.text_input(
        "Your ID",
        value="usr_001",
        key="user_id",
        disabled=started,
        help="Your sessions are saved under this ID. Start a new conversation to change it.",
    )

    # Ollama status indicator
    try:
        import requests as _req
        r = _req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    status_color = "#48bb78" if ollama_ok else "#fc8181"
    status_text = "AI Engine Online" if ollama_ok else "AI Engine Offline"
    st.markdown(
        f'<div style="font-size:0.75rem; color:{c["muted"]}; margin-bottom:0.5rem;">'
        f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        f'background:{status_color};margin-right:6px;"></span>{status_text}</div>',
        unsafe_allow_html=True,
    )

    st.button("➕ New Conversation", key="new_conv_btn", on_click=reset_conversation, use_container_width=True)
    st.divider()

    # Chat history library
    st.subheader("📚 Your Conversations")
    chats = CHATS.list(current_user_id())
    if chats:
        for day, day_chats in groupby(chats, key=lambda ch: ch.get("date", "")):
            st.markdown(
                f'<p class="chat-item-date">📅 {day}</p>',
                unsafe_allow_html=True,
            )
            for ch in day_chats:
                col_open, col_del = st.columns([5, 1])
                label = f"{ch.get('time', '')} · {ch.get('title', 'Chat')}"
                col_open.button(
                    label,
                    key=f"open_{ch['id']}",
                    on_click=open_chat,
                    args=(ch["id"],),
                    use_container_width=True,
                    help="Open this conversation",
                )
                col_del.button(
                    "🗑️",
                    key=f"del_{ch['id']}",
                    on_click=delete_chat,
                    args=(ch["id"],),
                    help="Delete this conversation",
                )
    else:
        st.caption("No saved conversations yet. Start chatting above.")

    st.divider()

    # Resources & Safety
    st.subheader("🆘 Crisis Resources")
    with st.expander("India"):
        st.markdown(
            "📞 **Tele-MANAS**: 1800-891-4416 *(24/7)*\n\n"
            "📞 **KIRAN**: 1800-599-0019 *(24/7)*\n\n"
            "📞 **iCall (TISS)**: 9152987821"
        )
    with st.expander("International"):
        st.markdown(
            "📞 **988 Lifeline (USA)**: 988 *(24/7)*\n\n"
            "📞 **Samaritans (UK)**: 116 123 *(24/7)*\n\n"
            "🌐 **IASP**: iasp.info/resources/Crisis_Centres/"
        )

    st.divider()
    st.info(
        "⚠️ If you are in immediate danger, please call emergency services "
        "or a crisis helpline right now. This AI cannot provide emergency assistance.",
        icon="🚨",
    )

    st.caption("")
