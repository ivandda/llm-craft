import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional, List

def stable_id(parts: List[str]) -> str:
    # Standardize serialization of concept arrays to ensure consistent hashing across runs
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

@dataclass
class RecipeObservation:
    input_a: str
    input_b: str
    output: str
    emoji_a: Optional[str] = None
    emoji_b: Optional[str] = None
    emoji_output: Optional[str] = None
    source: str = ""

    def to_dict(self):
        return asdict(self)
