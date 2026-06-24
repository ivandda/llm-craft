import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QualityConfig:
    max_words: int = 3
    max_chars: int = 36
    min_token_chars: int = 2
    reject_digits: bool = True
    reject_gerund_phrases: bool = True
    reject_sentence_words: bool = True
    reject_vowelless_tokens_min_len: int = 4
    suspicious_suffixes: tuple[str, ...] = ("nado",)
    placeholder_tokens: tuple[str, ...] = ("undefined", "unknown", "null", "none")


@dataclass(frozen=True)
class ConceptQuality:
    keep: bool
    reason: str = "ok"


HARD_SYMBOL_RE = re.compile(r"[#`;{}\[\]|\\<>+*=]")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")
VOWEL_RE = re.compile(r"[aeiou]")
SENTENCE_WORDS = {
    "because",
    "inside",
    "instead",
    "that",
    "this",
    "when",
    "while",
    "with",
    "without",
    "your",
}


def quality_config_from_dict(raw_config: dict[str, Any] | None) -> QualityConfig:
    raw_config = raw_config or {}
    suffixes = raw_config.get("suspicious_suffixes", ["nado"])
    placeholders = raw_config.get("placeholder_tokens", ["undefined", "unknown", "null", "none"])
    return QualityConfig(
        max_words=int(raw_config.get("max_words", 3)),
        max_chars=int(raw_config.get("max_chars", 36)),
        min_token_chars=int(raw_config.get("min_token_chars", 2)),
        reject_digits=bool(raw_config.get("reject_digits", True)),
        reject_gerund_phrases=bool(raw_config.get("reject_gerund_phrases", True)),
        reject_sentence_words=bool(raw_config.get("reject_sentence_words", True)),
        reject_vowelless_tokens_min_len=int(raw_config.get("reject_vowelless_tokens_min_len", 4)),
        suspicious_suffixes=tuple(str(item).strip().lower() for item in suffixes if str(item).strip()),
        placeholder_tokens=tuple(str(item).strip().lower() for item in placeholders if str(item).strip()),
    )


def score_concept(text: str, config: QualityConfig) -> ConceptQuality:
    concept = normalize_concept(text)
    if not concept:
        return ConceptQuality(False, "empty")

    if len(concept) > config.max_chars:
        return ConceptQuality(False, "too_long")

    if HARD_SYMBOL_RE.search(concept):
        return ConceptQuality(False, "hard_symbol")

    if any(mark in concept for mark in (":", "?", "!", "\"", "(", ")")):
        return ConceptQuality(False, "title_or_sentence_like")

    tokens = TOKEN_RE.findall(concept)
    if not tokens:
        return ConceptQuality(False, "no_word_tokens")

    if len(tokens) > config.max_words:
        return ConceptQuality(False, "too_many_words")

    if concept.replace(".", "").replace(",", "").replace("-", "").replace("%", "").isdigit():
        return ConceptQuality(False, "numeric")

    if config.reject_digits and any(any(char.isdigit() for char in token) for token in tokens):
        return ConceptQuality(False, "has_digit")

    if len(tokens) != len(set(tokens)):
        return ConceptQuality(False, "repeated_token")

    if config.reject_gerund_phrases and len(tokens) > 1 and any(token.endswith("ing") for token in tokens[1:]):
        return ConceptQuality(False, "gerund_phrase")

    if config.reject_sentence_words and len(tokens) >= 3 and any(token in SENTENCE_WORDS for token in tokens):
        return ConceptQuality(False, "sentence_like")

    for token in tokens:
        if token in config.placeholder_tokens:
            return ConceptQuality(False, "placeholder_token")
        if len(token) < config.min_token_chars:
            return ConceptQuality(False, "short_token")
        if _has_suspicious_suffix(token, config):
            return ConceptQuality(False, "suspicious_suffix")
        if _is_vowelless_noise(token, config):
            return ConceptQuality(False, "vowelless_token")

    return ConceptQuality(True)


def recipe_quality_reason(input_a: str, input_b: str, output: str, config: QualityConfig) -> str:
    for label, concept in (("input_a", input_a), ("input_b", input_b), ("output", output)):
        concept_quality = score_concept(concept, config)
        if not concept_quality.keep:
            return f"{label}:{concept_quality.reason}"

    if output == input_a or output == input_b:
        return "identity"

    return "ok"


def normalize_concept(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _has_suspicious_suffix(token: str, config: QualityConfig) -> bool:
    return any(token.endswith(suffix) and len(token) > len(suffix) + 2 for suffix in config.suspicious_suffixes)


def _is_vowelless_noise(token: str, config: QualityConfig) -> bool:
    min_len = config.reject_vowelless_tokens_min_len
    if min_len <= 0 or len(token) < min_len:
        return False
    return token.isalpha() and VOWEL_RE.search(token) is None
