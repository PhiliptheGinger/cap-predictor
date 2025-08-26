from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import re
import yaml


@dataclass
class Intent:
    name: str
    examples: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    required_slots: List[str] = field(default_factory=list)
    slot_patterns: Dict[str, str] = field(default_factory=dict)


class Ontology:
    """Domain ontology loaded from ``intents.yaml``."""

    def __init__(self, path: Path):
        self.path = path
        raw = yaml.safe_load(path.read_text())
        self.intents: Dict[str, Intent] = {}
        for item in raw.get("intents", []):
            intent = Intent(
                name=item["name"],
                examples=item.get("examples", []),
                signals=item.get("signals", []),
                required_slots=item.get("required_slots", []),
                slot_patterns=item.get("slot_patterns", {}),
            )
            self.intents[intent.name] = intent
        # entity patterns available globally
        self.entities: Dict[str, re.Pattern[str]] = {}
        for name, pattern in raw.get("entities", {}).items():
            self.entities[name] = re.compile(pattern, re.IGNORECASE)

    def get_intent(self, name: str) -> Intent:
        return self.intents[name]

    def required_slots(self, intent: str) -> List[str]:
        return self.intents[intent].required_slots

    def signals(self, intent: str) -> List[str]:
        return self.intents[intent].signals

