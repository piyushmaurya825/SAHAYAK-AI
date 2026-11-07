"""
Main LangGraph multi-agent cognitive pipeline.
Implements all 7 layers as a compiled StateGraph.

Flow:
START -> memory_load -> intake -> safety_assessment
         |-> crisis_escalation (if crisis)
         |-> rag_retrieval -> compatibility_planning -> response_generation
             -> guardrail_check -> [retry loop] -> finalize_response
Both paths -> memory_save -> evaluation -> END
"""

from __future__ import annotations
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PRIMARY_MODEL, FAST_MODEL, OLLAMA_BASE_URL,
    SAFETY_RETRY_MAX, GENERATION_TEMPERATURE, DEFAULT_HELPLINE_REGION
)
from agent.state import AgentState

# Layer imports
from agent.intake.emotion_capture import detect_and_interpret
from agent.intake.cultural_background import extract_cultural_background
from agent.intake.religious_beliefs import detect_faith
from agent.intake.values_system import extract_values
from agent.intake.preferences_capture import detect_preferences
from agent.orchestration.safety_assessor import assess_safety, get_helplines
from agent.orchestration.compatibility_checker import check_compatibility
from agent.orchestration.response_planner import create_response_plan
from agent.retrieval.query_builder import build_domain_queries
from agent.retrieval.context_assembler import assemble_context
from agent.retrieval.local_resources_retriever import get_local_resources
from agent.generation.response_generator import generate_response
from agent.generation.guardrails import run_all_guardrails
from agent.escalation.crisis_triage import (
    triage_crisis, log_crisis_incident,
    simulate_professional_notification, compose_crisis_response
)
from agent.memory.profile_store import profile_store, conversation_store, metrics_store
from agent.evaluation.metrics import evaluate_session
from agent.evaluation.langfuse_logger import log_session_trace, log_guardrail_event



def _get_primary_llm():
    return ChatOllama(
        model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL,
        temperature=GENERATION_TEMPERATURE
    )

def _get_fast_llm():
    return ChatOllama(model=FAST_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)


def intake_node(state: AgentState) -> AgentState:
    """
    Layer 1 (Nodes 1.1-1.7): Parallel intake of all 5 user dimensions.
    Assembles the UserPackage and initializes session metadata.
    """
    user_message = state.get("raw_message", "")
    user_id = state.get("user_id", "usr_001")
    history = state.get("conversation_history", [])

    # Load existing profile for continuity
    existing_profile = profile_store.get(user_id)
    fast_llm = _get_fast_llm()

    # 5 parallel intake branches (sequential here due to single Ollama instance)
    cultural = extract_cultural_background(user_message, history, existing_profile, fast_llm)
    faith = detect_faith(user_message, fast_llm)
    values = extract_values(user_message, fast_llm)
    emotions = detect_and_interpret(user_message, fast_llm)
    prefs = detect_preferences(
        user_message,
        existing_profile.get("preferences") if existing_profile else None
    )

    user_package = {
        "cultural": cultural,
        "faith": faith,
        "values": values,
        "emotions": emotions,
        "preferences": prefs,
    }

    session_id = state.get("session_id") or f"sess_{uuid.uuid4().hex[:12]}"

    state["user_package"] = user_package
    state["intake_complete"] = True
    state["session_id"] = session_id
    state["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return state

def memory_load_node(state: AgentState) -> AgentState:
    """Node 2.4: Load long-term user memory from SQLite."""
    user_id = state.get("user_id", "usr_001")

    try:
        sessions = conversation_store.get_recent_sessions(user_id, limit=5)
        past_verses = conversation_store.get_past_verses(user_id)
        user_profile = profile_store.get(user_id)
        is_returning = bool(sessions)

        memory_context = None
        if sessions:
            last = sessions[0]
            memory_context = (
                f"Last session ({last['date']}): User felt "
                f"{last.get('primary_emotion', 'distress')} about "
                f"'{last.get('root_cause', 'their situation')[:100]}'. "
                f"Verse shown: {last.get('verse', 'None')}."
            )

        state["is_returning_user"] = is_returning
        state["memory_context"] = memory_context
        state["past_verses"] = past_verses
        state["user_profile"] = user_profile
    except Exception as e:
        print(f"[MemoryLoad] Error: {e}")
        state["is_returning_user"] = False
        state["memory_context"] = None
        state["past_verses"] = []
        state["user_profile"] = None

    return state


def safety_assessment_node(state: AgentState) -> AgentState:
    """Node 2.6-2.7: Multi-tier safety risk assessment."""
    user_message = state.get("raw_message", "")
    risk = assess_safety(user_message)

    state["risk_assessment"] = risk
    state["is_crisis"] = risk.get("is_crisis", False)
    return state


def compatibility_planning_node(state: AgentState) -> AgentState:
    """Nodes 2.5 + 2.8: Cultural compatibility check and response planning."""
    user_message = state.get("raw_message", "")
    user_package = state.get("user_package", {})
    memory_context = state.get("memory_context")
    assembled_context = state.get("assembled_context", "")

    # Cultural compatibility check (ensure initialized)
    compat = user_package.get("compatibility")
    if not compat:
        compat = check_compatibility(user_package)
        user_package["compatibility"] = compat
    state["cultural_tags"] = user_package.get("cultural", {}).get("cultural_tags", [])

    # Create 4-step response plan
    plan = create_response_plan(
        user_message, user_package, memory_context, assembled_context, _get_primary_llm()
    )
    state["response_plan"] = plan
    state["therapeutic_goal"] = plan.get("therapeutic_goal", "validation")
    return state



def rag_retrieval_node(state: AgentState) -> AgentState:
    """
    Nodes 3.1-3.5: Build queries, retrieve from all ChromaDB domains,
    filter by cultural relevance, and assemble context.
    """
    user_message = state.get("raw_message", "")
    user_package = state.get("user_package", {})
    past_verses = state.get("past_verses", [])
    
    # Ensure compatibility is checked before filtering domains
    compat = user_package.get("compatibility")
    if not compat:
        compat = check_compatibility(user_package)
        user_package["compatibility"] = compat
        state["cultural_tags"] = user_package.get("cultural", {}).get("cultural_tags", [])

    # Build specialized sub-queries for each domain
    queries = build_domain_queries(user_message, user_package, _get_fast_llm())

    retrieved: Dict[str, List[Dict]] = {}

    try:
        from knowledge_base.vector_store import get_vector_store
        vs = get_vector_store()

        if vs.is_fully_ready():
            culture_filter = compat.get("culture_filter", "Universal")

            if compat.get("use_spiritual", True):
                spiritual_chunks = vs.query_spiritual(
                    queries.get("spiritual", user_message),
                    culture_filter=culture_filter
                )
                # Skip already-shown verses for variety
                spiritual_chunks = [
                    c for c in spiritual_chunks
                    if c.get("metadata", {}).get("id") not in past_verses
                ]
                retrieved["spiritual"] = spiritual_chunks[:3]

            if compat.get("use_cbt", True):
                retrieved["cbt"] = vs.query_cbt(queries.get("cbt", user_message))

            if compat.get("use_cultural_practices", True):
                retrieved["cultural"] = vs.query_cultural(
                    queries.get("cultural", user_message),
                    culture_filter=culture_filter
                )

            retrieved["psychoedu"] = vs.query_psychoedu(
                queries.get("psychoedu", user_message)
            )

            # Query guided therapeutic responses (exemplars for exact response style)
            guided_chunks = vs.query_guided(queries.get("guided", user_message), n=2)
            if guided_chunks:
                retrieved["guided_responses"] = guided_chunks
        else:
            print("[RAG] Vector store not ready — using fallback knowledge.")
            retrieved = _get_fallback_knowledge(user_package)

    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")
        retrieved = _get_fallback_knowledge(user_package)

    # Get local support resources
    region = user_package.get("cultural", {}).get("region", "India")
    local_resources = get_local_resources(region=region)

    # Assemble structured context string for LLM
    assembled = assemble_context(retrieved, local_resources)

    state["retrieved_context"] = retrieved
    state["assembled_context"] = assembled
    return state


def _get_fallback_knowledge(user_package: Dict) -> Dict:
    """Fallback static knowledge when ChromaDB is unavailable."""
    return {
        "guided_responses": [{
            "text": (
                "Topic: Anxiety and Overwhelm\n"
                "Support strategy: Recognize strengths, ground in the present, and take one small step.\n"
                "Guided response: Feeling overwhelmed can make your abilities hard to see. "
                "Pause, take a few slow breaths, and focus on one manageable next step. "
                "You do not need to solve everything today."
            ),
            "metadata": {
                "id": "EXEMP_FALLBACK",
                "category": "Anxiety and Self-Doubt",
                "support_strategy": "Recognize strengths and build confidence",
                "guided_response": (
                    "Feeling overwhelmed can make your abilities hard to see. Pause, take a few slow breaths, "
                    "and focus on one manageable next step. You do not need to solve everything today."
                ),
            },
            "relevance": 0.8,
        }],
        "spiritual": [{
            "text": (
                "Bhagavad Gita 6.5: Let a man lift himself by his own self alone; "
                "let him not lower himself. For self alone is the friend of self, "
                "and self alone is the enemy of self. "
                "Concept: The mind as friend or enemy (Manobuddhi). "
                "Psychology: Cognitive Defusion (ACT)."
            ),
            "metadata": {
                "id": "BG_6_5", "chapter": 6, "verse": 5,
                "concept": "Manobuddhi", "psychology": "ACT - Cognitive Defusion"
            },
            "relevance": 0.7,
        }],
        "cbt": [{
            "text": (
                "CBT Technique: Cognitive Reframing. "
                "Identify negative automatic thoughts, examine evidence for and against them, "
                "and generate a balanced alternative perspective. "
                "Especially effective for perfectionism and catastrophic thinking."
            ),
            "metadata": {"name": "Cognitive Reframing", "category": "cognitive"},
            "relevance": 0.7,
        }],
    }


def response_generation_node(state: AgentState) -> AgentState:
    """
    Node 5.1: Generate therapeutic response using primary LLM.
    On retry, passes critique notes from failed guardrails.
    """
    user_message = state.get("raw_message", "")
    user_package = state.get("user_package", {})
    response_plan = state.get("response_plan", {})
    assembled_context = state.get("assembled_context", "")
    conversation_history = state.get("conversation_history", [])
    guardrail_report = state.get("guardrail_report") or {}
    retry_count = state.get("retry_count", 0)

    critique_notes = guardrail_report.get("critique_notes", []) if retry_count > 0 else None

    candidate = generate_response(
        user_message=user_message,
        user_package=user_package,
        response_plan=response_plan,
        assembled_context=assembled_context,
        conversation_history=conversation_history,
        critique_notes=critique_notes,
        llm=_get_primary_llm(),
    )

    state["candidate_response"] = candidate
    return state


def guardrail_check_node(state: AgentState) -> AgentState:
    """Nodes 5.2-5.3: Run all 6 guardrail checks concurrently."""
    candidate = state.get("candidate_response", "")
    user_message = state.get("raw_message", "")
    assembled_context = state.get("assembled_context", "")
    user_package = state.get("user_package", {})

    report = run_all_guardrails(
        response=candidate,
        user_message=user_message,
        assembled_context=assembled_context,
        user_package=user_package,
    )

    # Log guardrail results to Langfuse
    log_guardrail_event(
        state.get("session_id", "unknown"),
        report,
        state.get("retry_count", 0)
    )

    state["guardrail_report"] = report
    if not report.get("all_passed", True):
        state["retry_count"] = state.get("retry_count", 0) + 1
    return state


def finalize_response_node(state: AgentState) -> AgentState:
    """Node 4.0: Append medical disclaimer and finalize for delivery."""
    candidate = state.get("candidate_response", "")

    disclaimer = (
        "\n\n---\n"
        "*\U0001f499 This response is from an AI companion and does not replace "
        "professional psychiatric care, psychotherapy, or emergency services. "
        "If you are in immediate danger, please call emergency services.*"
    )

    state["final_response"] = candidate + disclaimer
    state["response_kind"] = "normal"
    return state



def crisis_escalation_node(state: AgentState) -> AgentState:
    """Nodes 6.1-6.3: Crisis triage, notification simulation, crisis response."""
    risk_assessment = state.get("risk_assessment", {})
    user_id = state.get("user_id", "usr_001")
    user_package = state.get("user_package") or {}
    region = user_package.get("cultural", {}).get("region", "India")

    # Map cultural region to helpline region
    if "Asian" in region or region in ["South Asian", "India"]:
        helpline_region = "India"
    elif "American" in region and "Latin" not in region:
        helpline_region = "USA"
    elif "European" in region:
        helpline_region = "UK"
    else:
        helpline_region = DEFAULT_HELPLINE_REGION

    # C-SSRS-based triage
    triage = triage_crisis(risk_assessment)

    # Log crisis incident (anonymized)
    incident_id = log_crisis_incident(user_id, risk_assessment, triage)

    # Simulated professional notification
    if triage.get("notify_professional"):
        simulate_professional_notification(
            incident_id, risk_assessment.get("severity", "MODERATE")
        )

    # Compose deterministic (non-LLM) crisis response
    crisis_response = compose_crisis_response(triage, risk_assessment, helpline_region)

    state["final_response"] = crisis_response
    state["response_kind"] = "crisis"
    state["crisis_notified"] = True
    state["crisis_log_id"] = incident_id
    state["helpline_contacts"] = get_helplines(helpline_region)
    return state



def memory_save_node(state: AgentState) -> AgentState:
    """Nodes 4.1-4.4: Persist session data and update user profile in SQLite."""
    user_id = state.get("user_id", "usr_001")
    session_id = state.get("session_id", f"sess_{uuid.uuid4().hex[:8]}")
    user_package = state.get("user_package") or {}
    is_crisis = state.get("is_crisis", False)
    response_kind = state.get("response_kind", "normal")

    try:
        retrieved = state.get("retrieved_context", {})
        spiritual_chunks = retrieved.get("spiritual", [])
        cbt_chunks = retrieved.get("cbt", [])

        verse_shown = (
            spiritual_chunks[0].get("metadata", {}).get("id", "")
            if spiritual_chunks else ""
        )
        cbt_shown = (
            cbt_chunks[0].get("metadata", {}).get("name", "")
            if cbt_chunks else ""
        )
        primary_emotion = user_package.get("emotions", {}).get("primary_emotion", "neutral")

        conversation_store.save_session(
            user_id=user_id,
            session_id=session_id,
            root_cause=state.get("raw_message", ""),
            primary_emotion=primary_emotion,
            verse=verse_shown,
            cbt=cbt_shown,
            is_crisis=is_crisis,
            response_kind=response_kind,
        )

        # Update long-term user profile
        cultural = user_package.get("cultural", {})
        faith = user_package.get("faith", {})
        prefs = user_package.get("preferences", {})

        profile_store.upsert(user_id, {
            "region": cultural.get("region", "Universal"),
            "community": cultural.get("community", "Unknown"),
            "faith": faith.get("faith", "Spiritual (non-religious)"),
            "spirituality_level": faith.get("spirituality_level", "moderate"),
            "openness_to_spiritual": faith.get("openness_to_spiritual_counselling", True),
            "preferred_tone": prefs.get("tone", "empathetic"),
            "preferred_modality": prefs.get("modality", "balanced"),
            "language_style": prefs.get("language_style", "formal English"),
            "collectivism_index": cultural.get("collectivism_index", 0.5),
        })

    except Exception as e:
        print(f"[MemorySave] Error: {e}")

    return state



def evaluation_node(state: AgentState) -> AgentState:
    """Nodes 7.1-7.2: Compute session metrics, log to SQLite and Langfuse."""
    try:
        user_message = state.get("raw_message", "")
        response = state.get("final_response", "")
        assembled_context = state.get("assembled_context", "")
        user_package = state.get("user_package") or {}
        guardrail_report = state.get("guardrail_report") or {}
        session_id = state.get("session_id", "unknown")
        user_id = state.get("user_id", "usr_001")

        # Crisis path uses deterministic safety metrics
        if state.get("response_kind") == "crisis":
            state["session_metrics"] = {
                "safety_score": 1.0,
                "empathy_score": 4.5,
                "cultural_relevance_score": 3.0,
                "grounding_score": 1.0,
                "audit_trail": "Crisis escalation path — deterministic response used.",
            }
            return state

        metrics = evaluate_session(
            user_message, response, assembled_context,
            user_package, guardrail_report
        )
        state["session_metrics"] = metrics

        # Persist to SQLite
        metrics_store.save(session_id, user_id, metrics)

        # Log full trace to Langfuse (if configured)
        log_session_trace(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
            response=response,
            metrics=metrics,
            metadata={
                "is_crisis": state.get("is_crisis", False),
                "retry_count": state.get("retry_count", 0),
                "cultural_region": user_package.get("cultural", {}).get("region", "Unknown"),
                "primary_emotion": user_package.get("emotions", {}).get("primary_emotion", "unknown"),
                "therapeutic_goal": state.get("therapeutic_goal", "validation"),
            }
        )

    except Exception as e:
        print(f"[Evaluation] Error: {e}")
        state["session_metrics"] = {}

    return state



def route_after_safety(state: AgentState) -> str:
    """Node 2.7 decision: Crisis path or normal RAG flow."""
    return "crisis_escalation" if state.get("is_crisis", False) else "rag_retrieval"


def route_after_guardrails(state: AgentState) -> str:
    """Node 5.4 decision: Retry generation or finalize response."""
    report = state.get("guardrail_report") or {}
    retry_count = state.get("retry_count", 0)

    if report.get("all_passed", True):
        return "finalize_response"

    if retry_count <= SAFETY_RETRY_MAX:
        print(f"[Guardrails] Retry {retry_count}/{SAFETY_RETRY_MAX}: "
              f"{report.get('critique_notes', [])}")
        return "response_generation"

    # Max retries: finalize anyway (safe fallback content is in generator)
    print(f"[Guardrails] Max retries ({SAFETY_RETRY_MAX}) exceeded. Finalizing.")
    return "finalize_response"



def build_mental_health_agent():
    """
    Build and compile the full 7-layer LangGraph multi-agent pipeline.
    Returns a compiled CompiledGraph ready for .invoke() or .stream().
    """
    workflow = StateGraph(AgentState)

    # Register all nodes
    workflow.add_node("memory_load", memory_load_node)
    workflow.add_node("intake", intake_node)
    workflow.add_node("safety_assessment", safety_assessment_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("compatibility_planning", compatibility_planning_node)
    workflow.add_node("response_generation", response_generation_node)
    workflow.add_node("guardrail_check", guardrail_check_node)
    workflow.add_node("finalize_response", finalize_response_node)
    workflow.add_node("crisis_escalation", crisis_escalation_node)
    workflow.add_node("memory_save", memory_save_node)
    workflow.add_node("evaluation", evaluation_node)

    # Pipeline edges
    workflow.set_entry_point("memory_load")
    workflow.add_edge("memory_load", "intake")
    workflow.add_edge("intake", "safety_assessment")

    # Safety gate (crisis or normal)
    workflow.add_conditional_edges(
        "safety_assessment",
        route_after_safety,
        {
            "crisis_escalation": "crisis_escalation",
            "rag_retrieval": "rag_retrieval",
        }
    )

    # Normal flow: RAG -> Planning -> Generation -> Guardrails
    workflow.add_edge("rag_retrieval", "compatibility_planning")
    workflow.add_edge("compatibility_planning", "response_generation")
    workflow.add_edge("response_generation", "guardrail_check")

    # Guardrail gate (retry loop or finalize)
    workflow.add_conditional_edges(
        "guardrail_check",
        route_after_guardrails,
        {
            "response_generation": "response_generation",
            "finalize_response": "finalize_response",
        }
    )

    # Convergence: both paths -> memory -> evaluation -> END
    workflow.add_edge("finalize_response", "memory_save")
    workflow.add_edge("crisis_escalation", "memory_save")
    workflow.add_edge("memory_save", "evaluation")
    workflow.add_edge("evaluation", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def make_initial_state(
    user_id: str,
    message: str,
    conversation_history: Optional[List[Dict]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a fresh AgentState dict for a new conversation turn."""
    return {
        "user_id": user_id,
        "session_id": session_id or f"sess_{uuid.uuid4().hex[:12]}",
        "thread_id": "",
        "timestamp": "",
        "raw_message": message,
        "conversation_history": conversation_history or [],
        "user_package": None,
        "intake_complete": False,
        "risk_assessment": None,
        "response_plan": None,
        "cultural_tags": [],
        "therapeutic_goal": "",
        "is_crisis": False,
        "retrieved_context": {},
        "assembled_context": "",
        "is_returning_user": False,
        "memory_context": None,
        "past_verses": [],
        "user_profile": None,
        "candidate_response": "",
        "final_response": "",
        "guardrail_report": None,
        "retry_count": 0,
        "crisis_notified": False,
        "crisis_log_id": None,
        "session_metrics": None,
        "helpline_contacts": None,
        "response_kind": "normal",
    }
