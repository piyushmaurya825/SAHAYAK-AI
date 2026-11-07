"""
Defines all Pydantic schemas and the master AgentState TypedDict used
throughout the LangGraph pipeline.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import time



class CulturalBackground(BaseModel):
    region: str = Field(default="Universal", description="User's cultural/geographic region")
    community: str = Field(default="Unknown", description="Specific community or ethnicity")
    family_structure: str = Field(default="Unknown", description="Collectivist vs individualist")
    collectivism_index: float = Field(default=0.5, ge=0.0, le=1.0,
                                       description="0=individualist, 1=collectivist")
    cultural_tags: List[str] = Field(default_factory=list,
                                      description="Normalized cultural markers for RAG filtering")


class ReligiousBeliefs(BaseModel):
    faith: str = Field(default="Spiritual (non-religious)")
    openness_to_spiritual_counselling: bool = Field(default=True)
    spirituality_level: str = Field(default="moderate",
                                     description="strict|moderate|casual|secular")


class ValuesSystem(BaseModel):
    core_values: List[str] = Field(default_factory=list,
                                    description="e.g. ['duty', 'family honor', 'karma']")
    central_conflict: str = Field(default="",
                                   description="Primary value tension identified")


class EmotionalState(BaseModel):
    primary_emotion: str = Field(default="neutral")
    secondary_emotions: List[str] = Field(default_factory=list)
    intensity: float = Field(default=5.0, ge=0.0, le=10.0)
    valence: str = Field(default="neutral", description="positive|negative|mixed|neutral")
    hf_scores: Dict[str, float] = Field(default_factory=dict,
                                         description="Raw HuggingFace emotion scores")


class UserPreferences(BaseModel):
    tone: str = Field(default="empathetic", description="empathetic|direct|philosophical|gentle")
    modality: str = Field(default="balanced",
                           description="practical|reflective|spiritual|balanced")
    language_style: str = Field(default="formal English",
                                  description="Hinglish|formal English|casual|etc.")
    wants_practical_steps: bool = Field(default=True)
    wants_spiritual_guidance: bool = Field(default=True)


class UserPackage(BaseModel):
    cultural: CulturalBackground = Field(default_factory=CulturalBackground)
    faith: ReligiousBeliefs = Field(default_factory=ReligiousBeliefs)
    values: ValuesSystem = Field(default_factory=ValuesSystem)
    emotions: EmotionalState = Field(default_factory=EmotionalState)
    preferences: UserPreferences = Field(default_factory=UserPreferences)



class RiskAssessment(BaseModel):
    is_crisis: bool = Field(default=False)
    severity: str = Field(default="NONE", description="NONE|LOW|MODERATE|ACUTE")
    triggers: List[str] = Field(default_factory=list)
    c_ssrs_level: int = Field(default=0, ge=0, le=5,
                               description="Columbia Suicide Severity Rating Scale 0-5")
    moderation_flags: Dict[str, float] = Field(default_factory=dict)


class ResponsePlan(BaseModel):
    empathy_validation: str = Field(default="")
    cultural_framing: str = Field(default="")
    psychological_integration: str = Field(default="")
    micro_action: str = Field(default="")
    therapeutic_goal: str = Field(default="")
    selected_knowledge_domains: List[str] = Field(default_factory=list)


class GuardrailReport(BaseModel):
    harm_passed: bool = Field(default=True)
    cultural_sensitivity_passed: bool = Field(default=True)
    bias_passed: bool = Field(default=True)
    inclusiveness_passed: bool = Field(default=True)
    privacy_passed: bool = Field(default=True)
    faithfulness_passed: bool = Field(default=True)
    critique_notes: List[str] = Field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all([
            self.harm_passed, self.cultural_sensitivity_passed,
            self.bias_passed, self.inclusiveness_passed,
            self.privacy_passed, self.faithfulness_passed
        ])



class SessionMetrics(BaseModel):
    cultural_relevance_score: float = Field(default=0.0, ge=0.0, le=5.0)
    empathy_score: float = Field(default=0.0, ge=0.0, le=5.0)
    safety_score: float = Field(default=0.0, ge=0.0, le=1.0)
    grounding_score: float = Field(default=0.0, ge=0.0, le=1.0)
    faithfulness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    user_feedback: Optional[int] = Field(default=None, description="1-5 star rating")
    audit_trail: str = Field(default="", description="Human-readable decision trace")



class AgentState(TypedDict, total=False):
    # --- Session identifiers ---
    user_id: str
    session_id: str
    thread_id: str
    timestamp: str

    # --- Raw input ---
    raw_message: str
    conversation_history: List[Dict[str, str]]

    # --- Layer 1: User Package ---
    user_package: Optional[Dict]        
    intake_complete: bool

    # --- Layer 2: Orchestration ---
    risk_assessment: Optional[Dict]     
    response_plan: Optional[Dict]       
    cultural_tags: List[str]
    therapeutic_goal: str
    is_crisis: bool

    # --- Layer 3: RAG Context ---
    retrieved_context: Dict[str, List[Dict]]   
    assembled_context: str                      

    # --- Layer 4: Memory ---
    is_returning_user: bool
    memory_context: Optional[str]
    past_verses: List[str]
    user_profile: Optional[Dict]

    # --- Layer 5: Generation ---
    candidate_response: str
    final_response: str
    guardrail_report: Optional[Dict]    
    retry_count: int

    # --- Layer 6: Crisis Escalation ---
    crisis_notified: bool
    crisis_log_id: Optional[str]

    # --- Layer 7: Evaluation ---
    session_metrics: Optional[Dict]    

    # --- UI Metadata ---
    helpline_contacts: Optional[List[Dict]]
    response_kind: str                  
