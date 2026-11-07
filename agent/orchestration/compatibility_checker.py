"""
Validates cultural/faith compatibility before selecting knowledge domains.
"""

from __future__ import annotations
from typing import Any, Dict, List

def check_compatibility(user_package: Dict[str, Any]) -> Dict[str, Any]:

    faith = user_package.get("faith", {})
    cultural = user_package.get("cultural", {})
    prefs = user_package.get("preferences", {})

    faith_name = faith.get("faith", "Spiritual (non-religious)")
    openness = faith.get("openness_to_spiritual_counselling", True)
    spirituality = faith.get("spirituality_level", "moderate")
    wants_philosophical = faith.get("wants_philosophical_framing", True)
    region = cultural.get("region", "Universal")

    # Decide which spiritual corpus to prioritize
    spiritual_corpus_priority = []
    if openness and spirituality != "secular":
        if faith_name in ["Hindu", "Spiritual (non-religious)", "Other"]:
            spiritual_corpus_priority = ["Bhagavad Gita", "Upanishads", "Stoic"]
        elif faith_name in ["Buddhist"]:
            spiritual_corpus_priority = ["Buddhist mindfulness", "Dhammapada"]
        elif faith_name in ["Christian"]:
            spiritual_corpus_priority = ["Universal philosophical wisdom"]
        elif faith_name in ["Muslim"]:
            spiritual_corpus_priority = ["Universal philosophical wisdom"]
        else:
            spiritual_corpus_priority = ["Universal philosophical wisdom"]
    
    return {
        "use_spiritual": openness and spirituality != "secular",
        "use_cbt": True,  # Always available
        "use_cultural_practices": True,
        "use_local_resources": True,
        "use_psychoedu": True,
        "spiritual_corpus_priority": spiritual_corpus_priority,
        "culture_filter": region,
        "avoid_religious_projection": spirituality == "secular" or not openness,
        "collectivist_framing": cultural.get("collectivism_index", 0.5) > 0.6,
    }
