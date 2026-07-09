"""Loader del dataset {input_a, input_b, candidate_outputs:[...]}.

Cada candidato trae output, source (observed/teacher), rationale y rank (1 = mejor).
"""
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def load_candidates_jsonl(path: str) -> pd.DataFrame:
    """Aplana el JSONL a una fila por candidato (a, b, c, rank, source, rationale, weight).

    weight = 1/rank, así el candidato rank=1 pesa 1.0 y los de rank alto menos.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            a, b = obj["input_a"], obj["input_b"]
            for cand in obj["candidate_outputs"]:
                rows.append(
                    {
                        "a": a,
                        "b": b,
                        "c": cand["output"],
                        "rank": cand["rank"],
                        "source": cand["source"],
                        "rationale": cand.get("rationale", ""),
                        "weight": 1.0 / cand["rank"],
                    }
                )
    return pd.DataFrame(rows)


@dataclass
class KnownCandidateIndex:
    """Índice (a, b) -> candidatos conocidos con sus embeddings, para chequear si un
    output generado coincide con una respuesta ya presente para ese par."""
    lookup: dict = field(default_factory=dict)  # (a,b) -> list of dicts {c, rank, source, emb}

    @classmethod
    def build(cls, df: pd.DataFrame, embeddings: np.ndarray) -> "KnownCandidateIndex":
        """embeddings alineado fila a fila con df, correspondiente a la columna 'c'."""
        lookup = {}
        for i, row in df.reset_index(drop=True).iterrows():
            key = (row["a"].strip().lower(), row["b"].strip().lower())
            lookup.setdefault(key, []).append(
                {"c": row["c"], "rank": row["rank"], "source": row["source"], "emb": embeddings[i]}
            )
        return cls(lookup=lookup)

    def best_match(self, a: str, b: str, c_emb: np.ndarray) -> dict | None:
        """Candidato conocido más similar (coseno) a c_emb para el par (a, b), o None."""
        key = (a.strip().lower(), b.strip().lower())
        candidates = self.lookup.get(key)
        if not candidates:
            return None

        sims = [float(np.dot(c_emb, cand["emb"])) for cand in candidates]  # ya normalizados -> dot = cos_sim
        best_idx = int(np.argmax(sims))
        best = candidates[best_idx]
        return {
            "matched_c": best["c"],
            "matched_rank": best["rank"],
            "matched_source": best["source"],
            "similarity": sims[best_idx],
        }
