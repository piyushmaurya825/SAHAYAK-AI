"""
Looks up local mental health resources based on user region.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

DATA_FILE = Path(__file__).parent.parent.parent / "knowledge_base" / "data" / "local_resources.json"
_resources_cache = None

def _load_resources() -> List[Dict]:
    global _resources_cache
    if _resources_cache is None:
        try:
            _resources_cache = json.loads(DATA_FILE.read_text(encoding="utf-8-sig"))
        except Exception:
            _resources_cache = []
    return _resources_cache

def get_local_resources(region: str = "India", country: str = "India") -> List[Dict]:
    resources = _load_resources()
    # Filter by country first
    country_resources = [r for r in resources if r.get("country", "").lower() == country.lower()]
    if not country_resources:
        country_resources = [r for r in resources if r.get("country", "").lower() == "india"]
    # Prefer national helplines
    national = [r for r in country_resources if r.get("region", "").lower() in ["national", region.lower()]]
    return (national or country_resources)[:5]
