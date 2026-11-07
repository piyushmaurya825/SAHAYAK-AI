"""
C-SSRS-based crisis triage and simulated professional notification.
"""

from __future__ import annotations
import json
import uuid
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CRISIS_LOG, CRISIS_HELPLINES, DEFAULT_HELPLINE_REGION

GROUNDING_TECHNIQUES = """**Immediate Grounding Techniques:**
🌬️ **Box Breathing (4-4-4-4)**
   • Breathe IN for 4 counts → Hold for 4 → Breathe OUT for 4 → Hold for 4
   • Repeat 3-4 times

🖐️ **5-4-3-2-1 Sensory Grounding**
   • Name 5 things you can SEE right now
   • 4 things you can TOUCH
   • 3 things you can HEAR
   • 2 things you can SMELL
   • 1 thing you can TASTE"""


def triage_crisis(risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
    """Classify crisis severity and determine response protocol."""
    c_ssrs = risk_assessment.get("c_ssrs_level", 0)
    severity = risk_assessment.get("severity", "NONE")

    if c_ssrs >= 4 or severity == "ACUTE":
        return {
            "category": "IMMINENT_DANGER",
            "protocol": "emergency_referral",
            "message_template": "acute",
            "notify_professional": True,
        }
    elif c_ssrs >= 2 or severity == "MODERATE":
        return {
            "category": "ACTIVE_IDEATION",
            "protocol": "crisis_support",
            "message_template": "moderate",
            "notify_professional": True,
        }
    else:
        return {
            "category": "PASSIVE_DISTRESS",
            "protocol": "supportive_counselling",
            "message_template": "low",
            "notify_professional": False,
        }


def log_crisis_incident(user_id: str, risk_assessment: Dict, triage: Dict) -> str:
    """Log crisis incident with anonymized user ID."""
    incident_id = str(uuid.uuid4())[:8].upper()
    anonymous_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]

    entry = {
        "incident_id": incident_id,
        "user_hash": anonymous_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "severity": risk_assessment.get("severity", "UNKNOWN"),
        "c_ssrs_level": risk_assessment.get("c_ssrs_level", 0),
        "category": triage.get("category", "UNKNOWN"),
        "triggers": risk_assessment.get("triggers", []),
    }

    CRISIS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CRISIS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"[CrisisTriage] Incident logged: {incident_id} | Severity: {entry['severity']}")
    return incident_id


def simulate_professional_notification(incident_id: str, severity: str) -> None:
    """
    SIMULATED professional notification.
    In production: replace with Twilio SMS / SendGrid email / PagerDuty webhook.
    """
    notification = {
        "notification_type": "SIMULATED_ALERT",
        "incident_id": incident_id,
        "severity": severity,
        "timestamp": datetime.utcnow().isoformat(),
        "action": "Alert logged to crisis_log.jsonl. In production: Twilio/SendGrid would fire here.",
        "on_call_counselor": "Not configured (production would lookup on-call schedule)",
    }
    notif_file = CRISIS_LOG.parent / "notifications_log.jsonl"
    with open(notif_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(notification) + "\n")
    print(f"[CrisisTriage] SIMULATED notification sent for incident {incident_id}")


def compose_crisis_response(
    triage: Dict,
    risk_assessment: Dict,
    region: str = DEFAULT_HELPLINE_REGION
) -> str:
    """Generate a deterministic, safe crisis response message."""
    template = triage.get("message_template", "moderate")
    helplines = CRISIS_HELPLINES.get(region, CRISIS_HELPLINES[DEFAULT_HELPLINE_REGION])
    helpline_text = "\n".join(
        f"📞 **{h['name']}**: {h['number']} *(Available {h['available']})*"
        for h in helplines
    )

    if template == "acute":
        return (
            "💙 **I hear you. Your life matters deeply.**\n\n"
            "You've reached out, and that takes courage. Right now, I need to ask you to do one thing: "
            "**please contact a crisis professional immediately.** You deserve real human support right now.\n\n"
            f"**Reach out right now:**\n{helpline_text}\n\n"
            f"{GROUNDING_TECHNIQUES}\n\n"
            "---\n"
            "*I am an AI companion. I cannot provide emergency care, but these trained professionals can "
            "and will help you through this moment. Please call now.*"
        )
    elif template == "moderate":
        return (
            "💙 **I'm here with you, and I want you to know you don't have to carry this alone.**\n\n"
            "What you're feeling sounds incredibly heavy right now. These feelings are real, and they deserve "
            "real support — from a trained professional who can truly be there for you.\n\n"
            f"**Please consider reaching out to:**\n{helpline_text}\n\n"
            f"{GROUNDING_TECHNIQUES}\n\n"
            "---\n"
            "*This is an AI companion. While I can provide support and perspective, please connect with "
            "a mental health professional for the level of care you deserve.*"
        )
    else:
        return (
            "💙 **I can sense this is a difficult time for you.**\n\n"
            "Your wellbeing matters. If you ever feel unsafe or need immediate support, please reach out:\n\n"
            f"**Support resources:**\n{helpline_text}\n\n"
            "I'm here to listen and support you through this."
        )
