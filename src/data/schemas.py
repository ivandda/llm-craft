from dataclasses import dataclass, asdict
from typing import Optional

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
