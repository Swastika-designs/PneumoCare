"""T12: context-based word-sense disambiguation of clinical abbreviations.

Simplified Lesk algorithm: choose the sense whose *signature* (a bag of words describing
the sense) overlaps most with the words around the ambiguous abbreviation.

Three signature variants are compared:
  * ``lesk``           hand-written gloss words only (wsd_data.SENSES)
  * ``corpus_lesk``    gloss words + words that co-occur with the sense's full form in the
                       Medical Abstracts corpus (data-driven signature expansion)
  * ``corpus_only``    only the corpus-derived words. The glosses and the evaluation sentences
                       were written by the same person, so ``lesk`` scores are optimistic;
                       ``corpus_only`` is the fairer, fully data-driven estimate.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.core.errors import ValidationFailed
from app.core.utils import to_jsonable
from app.services.wsd_data import EXAMPLES, SENSES

WORD = re.compile(r"[a-z][a-z0-9-]+")


def _tokens(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS and len(w) > 2]


def _stem(word: str) -> str:
    """Crude suffix stripping so 'lesions'/'lesion' and 'anticoagulation'/'anticoagulant' match."""
    for suffix in ("ation", "ions", "ion", "ing", "ed", "es", "s", "al", "ic"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


@lru_cache(maxsize=1)
def corpus_signatures(window: int = 12, top_n: int = 25) -> dict[str, dict[str, list[str]]]:
    """Expand each sense with words that frequently surround its full form in the corpus."""
    try:
        from app.services.text_analytics import load_corpus

        train, _, _ = load_corpus()
        texts = train["medical_abstract"].str.lower().tolist()
    except Exception:
        return {}
    background = Counter(w for t in texts[:3000] for w in set(_tokens(t)))
    tokenized = [WORD.findall(t) for t in texts if any(sense in t for senses in SENSES.values()
                                                         for sense in senses)]
    expanded: dict[str, dict[str, list[str]]] = {}
    for abbr, senses in SENSES.items():
        expanded[abbr] = {}
        for sense in senses:
            phrase = sense.split()
            n = len(phrase)
            counts: Counter = Counter()
            for words in tokenized:
                for i in range(len(words) - n + 1):
                    if words[i:i + n] == phrase:
                        context = words[max(0, i - window):i] + words[i + n:i + n + window]
                        counts.update(w for w in context if w not in ENGLISH_STOP_WORDS and len(w) > 2
                                      and w not in phrase)
            # Prefer words that are specific to the sense rather than common everywhere.
            scored = {w: c / (1 + background.get(w, 0) / 30) for w, c in counts.items() if c >= 2}
            expanded[abbr][sense] = [w for w, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:top_n]]
    return expanded


VARIANTS = ("lesk", "corpus_lesk", "corpus_only")


def _signature(abbr: str, sense: str, variant: str) -> set[str]:
    words = set() if variant == "corpus_only" else set(_tokens(SENSES[abbr][sense] + " " + sense))
    if variant in {"corpus_lesk", "corpus_only"}:
        words |= set(corpus_signatures().get(abbr, {}).get(sense, []))
    return {_stem(w) for w in words}


def disambiguate(sentence: str, abbreviation: str | None = None, variant: str = "corpus_lesk") -> dict:
    if variant not in VARIANTS:
        raise ValidationFailed(f"variant must be one of {', '.join(VARIANTS)}")
    if not sentence or not sentence.strip():
        raise ValidationFailed("Enter a sentence containing an abbreviation.")
    if abbreviation is None:
        found = [a for a in SENSES if re.search(rf"\b{a}\b", sentence)]
        if not found:
            raise ValidationFailed(f"No supported abbreviation found. Supported: {', '.join(SENSES)}")
        abbreviation = found[0]
    abbreviation = abbreviation.upper()
    if abbreviation not in SENSES:
        raise ValidationFailed(f"Unsupported abbreviation '{abbreviation}'. Supported: {', '.join(SENSES)}")
    context = [_stem(w) for w in _tokens(re.sub(rf"\b{abbreviation}\b", " ", sentence))]
    context_set = set(context)
    scores = []
    for sense in SENSES[abbreviation]:
        signature = _signature(abbreviation, sense, variant)
        overlap = sorted(context_set & signature)
        scores.append({"sense": sense, "score": len(overlap), "overlap": overlap})
    # Ties are broken by inventory order (the first listed sense acts as the default).
    best = max(scores, key=lambda s: s["score"])
    return to_jsonable({
        "sentence": sentence,
        "abbreviation": abbreviation,
        "variant": variant,
        "context_words": sorted(context_set),
        "scores": scores,
        "predicted_sense": best["sense"],
        "confident": best["score"] > 0 and sum(s["score"] == best["score"] for s in scores) == 1,
    })


def evaluate() -> dict:
    """Accuracy of both variants (and a most-frequent-sense baseline) on the labelled examples."""
    rows, summary = [], {}
    predictions = {v: [disambiguate(s, a, v)["predicted_sense"] for a, _, s in EXAMPLES] for v in VARIANTS}
    for variant in VARIANTS:
        summary[variant] = sum(p == gold for p, (_, gold, _) in zip(predictions[variant], EXAMPLES)) / len(EXAMPLES)
    for i, (abbr, gold, sentence) in enumerate(EXAMPLES):
        rows.append({"abbreviation": abbr, "sentence": sentence, "gold": gold,
                     **{f"pred_{v}": predictions[v][i] for v in VARIANTS},
                     "correct": predictions["corpus_lesk"][i] == gold})
    summary["first_sense_baseline"] = sum(gold == next(iter(SENSES[abbr])) for abbr, gold, _ in EXAMPLES) \
        / len(EXAMPLES)
    return to_jsonable({
        "examples": len(EXAMPLES),
        "accuracy": summary,
        "predictions": rows,
        "inventory": {abbr: list(senses) for abbr, senses in SENSES.items()},
        "data_note": "Evaluation sentences are hand-written synthetic examples (see wsd_data.py). The same author "
                     "wrote the glosses, so 'lesk' accuracy is optimistic; 'corpus_only' uses no hand-written "
                     "glosses.",
        "expanded_signatures": corpus_signatures(),
    })
