"""T11: clinical information extraction with spaCy.

Pipeline
  1. spaCy statistical model for tokens, sentences and general entities (DATE, QUANTITY, ...).
     Preference order: ``en_ner_bc5cdr_md`` (scispaCy DISEASE/CHEMICAL, optional) ->
     ``en_core_web_sm`` -> ``spacy.blank('en')``.
  2. An EntityRuler with a clinical lexicon and token patterns adds PROBLEM, MEDICATION,
     DOSAGE, VITAL, LAB, PROCEDURE and ANATOMY entities.
  3. NegEx-style rules mark entities as negated ("denies chest pain"), historical
     ("history of asthma") or family history.
"""
from __future__ import annotations

import threading
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.util import filter_spans

from app.core.errors import ValidationFailed
from app.core.utils import to_jsonable

PROBLEMS = [
    "pneumonia", "community acquired pneumonia", "asthma", "copd", "chronic obstructive pulmonary disease",
    "bronchitis", "tuberculosis", "pleural effusion", "pulmonary embolism", "atelectasis", "consolidation",
    "infiltrate", "hypertension", "type 2 diabetes", "diabetes mellitus", "diabetes", "hyperlipidemia",
    "coronary artery disease", "heart failure", "congestive heart failure", "atrial fibrillation",
    "myocardial infarction", "stroke", "sepsis", "anemia", "chronic kidney disease", "acute kidney injury",
    "obesity", "depression", "anxiety", "migraine", "epilepsy", "seizure", "multiple sclerosis", "cancer",
    "carcinoma", "lymphoma", "leukemia", "tumor", "neoplasm", "cirrhosis", "hepatitis", "pancreatitis",
    "gastroenteritis", "urinary tract infection", "cellulitis", "covid-19", "influenza", "otitis media",
    "fever", "cough", "productive cough", "shortness of breath", "dyspnea", "chest pain", "wheezing",
    "fatigue", "headache", "nausea", "vomiting", "diarrhea", "abdominal pain", "dizziness", "syncope",
    "edema", "rash", "chills", "night sweats", "weight loss", "hemoptysis", "tachycardia", "hypoxia",
    "palpitations", "sore throat", "back pain", "confusion",
]
MEDICATIONS = [
    "amoxicillin", "azithromycin", "ceftriaxone", "levofloxacin", "doxycycline", "vancomycin",
    "piperacillin-tazobactam", "metformin", "insulin", "lisinopril", "losartan", "amlodipine",
    "hydrochlorothiazide", "metoprolol", "atorvastatin", "simvastatin", "aspirin", "clopidogrel",
    "warfarin", "heparin", "apixaban", "albuterol", "ipratropium", "prednisone", "methylprednisolone",
    "fluticasone", "montelukast", "acetaminophen", "paracetamol", "ibuprofen", "morphine", "oxycodone",
    "furosemide", "omeprazole", "pantoprazole", "ondansetron", "sertraline", "fluoxetine", "levothyroxine",
    "gabapentin", "oseltamivir", "oxygen", "nitroglycerin", "digoxin",
]
LABS = [
    "white blood cell count", "wbc", "hemoglobin", "hba1c", "hemoglobin a1c", "creatinine", "bun",
    "glucose", "sodium", "potassium", "troponin", "procalcitonin", "c-reactive protein", "crp",
    "lactate", "bnp", "d-dimer", "platelets", "cholesterol", "ldl", "hdl", "blood culture",
    "sputum culture",
]
PROCEDURES = [
    "chest x-ray", "chest radiograph", "ct scan", "chest ct", "mri", "ultrasound", "echocardiogram",
    "ecg", "ekg", "electrocardiogram", "bronchoscopy", "intubation", "mechanical ventilation", "biopsy",
    "colonoscopy", "endoscopy", "surgery", "dialysis", "physical therapy", "blood transfusion",
]
ANATOMY = [
    "lung", "lungs", "left lower lobe", "right lower lobe", "right middle lobe", "left upper lobe",
    "right upper lobe", "chest", "heart", "liver", "kidney", "abdomen", "brain", "throat", "pleura",
]
UNITS = ["mg", "mcg", "g", "ml", "mL", "units", "iu", "IU", "mg/kg", "l/min", "L/min", "puffs", "tablets"]
FREQUENCIES = ["daily", "bid", "tid", "qid", "qd", "prn", "q4h", "q6h", "q8h", "q12h", "nightly", "weekly",
               "twice daily", "once daily", "three times daily"]

NEGATION_TRIGGERS = {"no", "not", "denies", "denied", "without", "negative", "absent", "free", "ruled",
                     "never", "nor"}
NEGATION_PHRASES = ["negative for", "ruled out", "rules out", "free of", "no evidence of", "no signs of",
                    "absence of", "resolved"]
TERMINATORS = {"but", "however", "although", "except", "though", "which", "who"}
HISTORY_PHRASES = ["history of", "h/o", "past medical history", "previous", "prior", "status post", "s/p"]
FAMILY_PHRASES = ["family history", "mother", "father", "brother", "sister", "maternal", "paternal"]

ENTITY_COLORS = {
    "PROBLEM": "#ef4444", "DISEASE": "#ef4444", "MEDICATION": "#3b82f6", "CHEMICAL": "#3b82f6",
    "DOSAGE": "#06b6d4", "VITAL": "#f59e0b", "LAB": "#8b5cf6", "PROCEDURE": "#10b981", "ANATOMY": "#ec4899",
    "DATE": "#64748b", "TIME": "#64748b", "QUANTITY": "#64748b", "CARDINAL": "#94a3b8", "PERSON": "#94a3b8",
    "AGE": "#64748b",
}

SAMPLE_NOTES = [
    {
        "id": "note-1",
        "title": "Emergency department note (synthetic)",
        "text": "65 year old male with a history of hypertension and type 2 diabetes presents with fever, "
                "productive cough and shortness of breath for 3 days. Denies chest pain or hemoptysis. "
                "Temp 38.9 C, HR 112, BP 138/84, RR 24, SpO2 91% on room air. Chest x-ray shows consolidation "
                "in the right lower lobe consistent with community acquired pneumonia. WBC 15.2, procalcitonin "
                "elevated. Started ceftriaxone 1 g daily and azithromycin 500 mg daily. Continue metformin 500 mg "
                "bid and lisinopril 10 mg daily. No evidence of pleural effusion.",
    },
    {
        "id": "note-2",
        "title": "Outpatient follow-up (synthetic)",
        "text": "Patient returns for follow-up of asthma. Reports occasional wheezing at night but no fever. "
                "Family history of COPD in father. Uses albuterol 2 puffs prn and fluticasone 2 puffs bid. "
                "Lungs clear on auscultation. HbA1c 6.1. Plan: continue current inhalers, repeat spirometry "
                "in 6 months.",
    },
    {
        "id": "note-3",
        "title": "Discharge summary (synthetic)",
        "text": "Admitted with sepsis secondary to urinary tract infection. Blood culture grew E. coli. "
                "Treated with piperacillin-tazobactam then switched to oral levofloxacin 750 mg daily. "
                "Creatinine improved from 2.1 to 1.2. Ruled out pulmonary embolism with chest CT. "
                "Discharged home in stable condition; no confusion, tolerating diet.",
    },
]

_nlp_lock = threading.Lock()


def _phrase_patterns(label: str, terms: list[str]) -> list[dict]:
    # String patterns are tokenised by spaCy itself, so "x-ray" matches the same 3 tokens as the text.
    return [{"label": label, "pattern": term} for term in terms]


def _token_patterns() -> list[dict]:
    num = {"LIKE_NUM": True}
    unit = {"LOWER": {"IN": [u.lower() for u in UNITS]}}
    freq = {"LOWER": {"IN": [f for f in FREQUENCIES if " " not in f]}, "OP": "?"}
    patterns = [
        {"label": "DOSAGE", "pattern": [num, unit, freq]},
        {"label": "DOSAGE", "pattern": [num, unit, {"LOWER": {"IN": ["twice", "once"]}}, {"LOWER": "daily"}]},
        {"label": "VITAL", "pattern": [{"LOWER": {"IN": ["bp", "blood"]}}, {"LOWER": "pressure", "OP": "?"},
                                       {"TEXT": {"REGEX": r"^\d{2,3}/\d{2,3}$"}}]},
        {"label": "VITAL", "pattern": [{"LOWER": {"IN": ["hr", "pulse", "rr"]}}, num]},
        {"label": "VITAL", "pattern": [{"LOWER": {"IN": ["temp", "temperature", "t"]}}, num,
                                       {"LOWER": {"IN": ["c", "f", "°c", "°f"]}, "OP": "?"}]},
        {"label": "VITAL", "pattern": [{"LOWER": {"IN": ["spo2", "o2", "sat", "saturation"]}}, num,
                                       {"TEXT": "%", "OP": "?"}]},
        {"label": "VITAL", "pattern": [{"LOWER": {"IN": ["spo2", "o2", "sat"]}}, {"TEXT": {"REGEX": r"^\d+%$"}}]},
        {"label": "AGE", "pattern": [num, {"LOWER": {"IN": ["year", "yr", "years"]}}, {"TEXT": "-", "OP": "?"},
                                     {"LOWER": "old"}]},
    ]
    for lab in LABS:
        toks = [{"LOWER": t} for t in lab.split()]
        patterns.append({"label": "LAB", "pattern": toks + [{"LOWER": {"IN": ["of", "is", "was"]}, "OP": "?"},
                                                             num]})
        patterns.append({"label": "LAB", "pattern": toks})
    return patterns


@lru_cache(maxsize=1)
def get_nlp() -> tuple[Language, str]:
    """Load the best available spaCy model and attach the clinical EntityRuler."""
    for name in ("en_ner_bc5cdr_md", "en_core_web_sm"):
        try:
            nlp = spacy.load(name)
            break
        except OSError:
            continue
    else:
        nlp, name = spacy.blank("en"), "blank:en"
        nlp.add_pipe("sentencizer")
    ruler_cfg = {"overwrite_ents": True, "phrase_matcher_attr": "LOWER"}
    ruler = nlp.add_pipe("entity_ruler", before="ner", config=ruler_cfg) if "ner" in nlp.pipe_names \
        else nlp.add_pipe("entity_ruler", config=ruler_cfg)
    ruler.add_patterns(
        _token_patterns()
        + _phrase_patterns("PROBLEM", PROBLEMS)
        + _phrase_patterns("MEDICATION", MEDICATIONS)
        + _phrase_patterns("PROCEDURE", PROCEDURES)
        + _phrase_patterns("ANATOMY", ANATOMY)
    )
    return nlp, name


def _context_flags(doc, ent) -> dict:
    sent = ent.sent
    window_start = max(sent.start, ent.start - 6)
    before = doc[window_start:ent.start]
    # Stop at terminators ("no fever but cough": 'cough' is not negated).
    tokens = list(before)
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i].lower_ in TERMINATORS or tokens[i].text in {";", "."}:
            tokens = tokens[i + 1:]
            break
    text_before = " ".join(t.lower_ for t in tokens)
    sentence_prefix = doc[sent.start:ent.start].text.lower()
    # The 6-token window lets lists inherit the cue: "denies chest pain or hemoptysis".
    negated = any(t.lower_ in NEGATION_TRIGGERS for t in tokens) or any(p in text_before for p in NEGATION_PHRASES)
    return {
        "negated": negated,
        "historical": any(p in text_before for p in HISTORY_PHRASES),
        "family": any(p in sentence_prefix for p in FAMILY_PHRASES),
    }


def extract(text: str) -> dict:
    if not text or not text.strip():
        raise ValidationFailed("Enter clinical text to analyse.")
    if len(text) > 20000:
        raise ValidationFailed("Text is too long (limit 20,000 characters).")
    nlp, model_name = get_nlp()
    with _nlp_lock:
        doc = nlp(text)
    ents = filter_spans(list(doc.ents))
    entities = []
    for ent in ents:
        if ent.label_ in {"ORDINAL", "NORP", "GPE", "LOC", "FAC", "ORG", "WORK_OF_ART", "LANGUAGE", "EVENT",
                          "LAW", "PRODUCT", "MONEY", "PERCENT"}:
            continue
        flags = _context_flags(doc, ent) if ent.label_ in {"PROBLEM", "DISEASE", "PROCEDURE", "MEDICATION",
                                                           "CHEMICAL"} else {"negated": False, "historical": False,
                                                                             "family": False}
        source = "rule" if ent.ent_id_ or ent.label_ in {"PROBLEM", "MEDICATION", "DOSAGE", "VITAL", "LAB",
                                                         "PROCEDURE", "ANATOMY", "AGE"} else "model"
        entities.append({"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char,
                         "sentence": next(i for i, s in enumerate(doc.sents) if s.start <= ent.start < s.end),
                         "source": source, **flags})
    # Link each medication to the closest following dosage in the same sentence.
    medications = []
    for i, e in enumerate(entities):
        if e["label"] in {"MEDICATION", "CHEMICAL"}:
            dose = next((d for d in entities[i + 1:i + 3] if d["label"] == "DOSAGE" and d["sentence"] == e["sentence"]),
                        None)
            medications.append({"medication": e["text"], "dosage": dose["text"] if dose else None,
                                "negated": e["negated"]})
    by_label: dict[str, list[str]] = {}
    for e in entities:
        by_label.setdefault(e["label"], [])
        if e["text"] not in by_label[e["label"]]:
            by_label[e["label"]].append(e["text"])
    structured = {
        "problems_present": sorted({e["text"].lower() for e in entities if e["label"] in {"PROBLEM", "DISEASE"}
                                    and not e["negated"] and not e["family"]}),
        "problems_negated": sorted({e["text"].lower() for e in entities if e["label"] in {"PROBLEM", "DISEASE"}
                                    and e["negated"]}),
        "family_history": sorted({e["text"].lower() for e in entities if e["family"]}),
        "medications": medications,
        "vitals": [e["text"] for e in entities if e["label"] == "VITAL"],
        "labs": [e["text"] for e in entities if e["label"] == "LAB"],
        "procedures": [e["text"] for e in entities if e["label"] == "PROCEDURE"],
    }
    return to_jsonable({
        "model": model_name,
        "pipeline": nlp.pipe_names,
        "text": text,
        "entities": entities,
        "by_label": by_label,
        "structured": structured,
        "sentences": [s.text for s in doc.sents],
        "colors": ENTITY_COLORS,
        "counts": {k: len(v) for k, v in by_label.items()},
    })


def corpus_entity_stats(per_class: int = 60, seed: int = 0) -> dict:
    """Run the extractor on a sample of abstracts per class and report the most frequent entities."""
    from app.services.text_analytics import load_corpus

    train, _, _ = load_corpus()
    nlp, model_name = get_nlp()
    sample = train.groupby("label", group_keys=False).apply(
        lambda g: g.sample(n=min(per_class, len(g)), random_state=seed))
    stats: dict[str, dict[str, dict[str, int]]] = {}
    with _nlp_lock:
        for label, doc in zip(sample["label"], nlp.pipe(sample["medical_abstract"].tolist(), batch_size=32)):
            for ent in doc.ents:
                if ent.label_ in {"PROBLEM", "DISEASE", "MEDICATION", "CHEMICAL", "PROCEDURE", "LAB"}:
                    bucket = stats.setdefault(label, {}).setdefault(ent.label_, {})
                    key = ent.text.lower()
                    bucket[key] = bucket.get(key, 0) + 1
    top = {label: {kind: sorted(terms.items(), key=lambda kv: -kv[1])[:8] for kind, terms in kinds.items()}
           for label, kinds in stats.items()}
    return to_jsonable({"model": model_name, "documents": len(sample), "per_class": per_class,
                        "top_entities": {label: {kind: [{"text": t, "count": c} for t, c in items]
                                                 for kind, items in kinds.items()} for label, kinds in top.items()}})
