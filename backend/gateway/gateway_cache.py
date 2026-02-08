from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    intent: str
    constraints: dict[str, Any]
    decision: dict[str, Any]
    created_at: str


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


class CacheStore:
    def __init__(self, path: Path, similarity_threshold: float = 0.9):
        self.path = path
        self.similarity_threshold = similarity_threshold

    def _load_entries(self) -> list[CacheEntry]:
        if not self.path.exists():
            return []
        entries: list[CacheEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                entries.append(CacheEntry(**raw))
            except Exception:
                continue
        return entries

    def _constraints_match(self, a: dict[str, Any], b: dict[str, Any]) -> bool:
        return a == b

    def find_match(self, intent: str, constraints: dict[str, Any]) -> CacheEntry | None:
        best_entry: CacheEntry | None = None
        best_score = 0.0
        for entry in self._load_entries():
            if not self._constraints_match(entry.constraints, constraints):
                continue
            score = _similarity(intent, entry.intent)
            if score >= self.similarity_threshold and score > best_score:
                best_entry = entry
                best_score = score
        return best_entry

    def write_entry(self, intent: str, constraints: dict[str, Any], decision: dict[str, Any]) -> CacheEntry:
        entry = CacheEntry(
            intent=intent,
            constraints=constraints,
            decision=decision,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(entry), ensure_ascii=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return entry
