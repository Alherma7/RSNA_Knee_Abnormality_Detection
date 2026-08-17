"""Weak-label extraction from free-text radiology reports.

Source: Ratner et al., "Data Programming: Creating Large Training Sets,
Quickly" (NeurIPS 2016) / Snorkel (snorkel.org). Why: with only a small
gold-labeled subset out of 4,407 studies, the report text is the only
supervision signal for the rest.

Critical schema fact (confirmed by both reviewed reference notebooks):
`train.csv` has a `Report` column, `test.csv` does not. Text exists only
at train time. This module's output is therefore a TRAINING LABEL SOURCE
only — never a model input feature at inference, since there would be
nothing to read at submission time (see src/model.py and src/features.py,
which take image-only input).

Two label sources are viable, per the reviewed notebooks:
1. A rule-based lexicon (this module) — cue-based, must test all
   REPORT_LANGUAGE_COUNT languages at once rather than routing by a
   guessed language (a cheap substring language guess was shown to fail,
   e.g. "la" is common to both Spanish and French).
2. A public LLM-derived label table, distributed as an attachable
   Kaggle Dataset by another notebook author (mentioned in
   pilkwang/rsna-knee-baseline-v1, section 2) — worth evaluating as an
   alternative or ensemble partner to (1) before committing to only one.

Validated against the 58-row gold subset in
notebooks/03_labeler_validation.ipynb (2026-08-17, run on both Kaggle and
locally against data/raw/, identical results both times): macro ROC-AUC
0.688 vs. a 0.5 constant-prediction baseline, every one of the 12
findings individually above 0.5 (weakest: oa_lateral_compartment at
0.553, driven by a high silence rate — see label_report() below). 58 rows
is a small validation set (as few as 9 positives for mcl_injury), so
these AUC numbers are noisy point estimates — treat further lexicon
tuning against this same fixed set with appropriate skepticism about
overfitting to it, per structuring-ml-projects's negative-result logging
guidance.
"""

import re
import unicodedata

import pandas as pd

# Both reference notebooks' "assertion, negation, hedge" clause reading —
# normalize (fold case/diacritics), unwrap (rejoin hard-wrapped lines),
# then split into clauses, matching cues against the union of all
# languages' vocabulary per clause (no language routing).


def _normalize(text: str) -> str:
    """Fold case, diacritics, and separators so cue regexes match either.

    NFKD decomposition strips accents (both Latin and other scripts'
    diacritics), which matters here because reports are inconsistent
    about them (e.g. "rotura" vs "rotúra").
    """
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[_\-/\\]+", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t


_BULLET_PREFIX_RE = re.compile(r"^[>*•\-]+\s*")


def _unwrap(text: str) -> str:
    """Rejoin lines that a fixed-width report layout broke mid-sentence.

    notebooks/02_eda_reports.ipynb section C measured this directly
    against all 4,407 reports (2026-08-17): only 23.2% have any
    wrap-candidate line pair (much less than "a large share of the
    corpus", the framing in the reviewed prvsiyan notebook this logic is
    adapted from) — so this is a real but minority phenomenon here, not
    the dominant one. The same EDA caught a false-positive in the
    original heuristic: a bulleted report (`> finding one`) was flagged
    as wrapped on every line, because a bullet character `>` is neither
    upper- nor lowercase, so `not s[:1].isupper()` was always true for
    it. Fixed here by stripping a leading bullet/quote marker before the
    capitalization check (but keeping it in the joined output).
    """
    if not isinstance(text, str):
        return ""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        s_check = _BULLET_PREFIX_RE.sub("", s)
        if (out and out[-1] and not re.search(r"[.;:!?>*•]$", out[-1])
                and len(out[-1].split()) >= 4 and s_check and not s_check[:1].isupper()):
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return "\n".join(out)


_SENT_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+|\n+")


def _clauses(text: str) -> "list[str]":
    """Normalize, unwrap, then split into sentence/line-level clauses."""
    norm = _normalize(_unwrap(text))
    return [c.strip() for c in _SENT_SPLIT_RE.split(norm) if c and c.strip()]


# Shared cartilage/joint-degeneration cues for the three osteoarthritis
# compartments — the same pathology vocabulary applies regardless of
# which compartment it's scoped to by the anatomy cue. Widened in
# notebooks/03_labeler_validation.ipynb section F after inspecting real
# gold-positive reports (initial narrower list had a 90%+ silence rate
# on oa_medial_compartment — radiologists use synonyms like "cartilage
# fissuring"/"spurring"/"subchondral cystic change", not just
# "osteoarthritis"/"osteophyte").
_OA_PATHOLOGY_CUES = [
    r"osteoarthrit", r"osteoarthros", r"osteoartr", r"chondrosis",
    r"(cartilage|chondral) (loss|thinning|defect|fissur)",
    r"joint space narrowing", r"osteophyte", r"osteofito", r"spurring",
    r"pinzamiento", r"degenerat", r"adelgazamiento del cartilago",
    r"chondromalacia", r"condromalacia", r"subchondral cystic",
]

# One entry per config.FINDINGS. `anatomy` cues identify that a clause is
# talking about this finding's structure at all; `pathology` cues
# identify an assertion of abnormality once scoped there. For findings
# where the entity name itself is the pathology (effusion, synovitis,
# baker's cyst, bone contusion — there's no separate "structure" to name
# before saying whether it's abnormal), `anatomy` and `pathology` are the
# same cue list by design: mentioning the entity without a negation cue
# nearby is itself the positive signal.
#
# English + Spanish only, in this first validated pass — the two
# languages that dominate the corpus per notebooks/02_eda_reports.ipynb
# section F (top-40 vocabulary). Coverage gaps in the other 7 languages
# (confirmed present via script detection, ~12% of reports by Greek +
# Cyrillic script alone) fall through to label_report()'s silent
# abstention rather than a wrong guess — see its docstring.
FINDING_LEXICON = {
    "acl_injury": dict(
        anatomy=[r"\bacl\b", r"anterior cruciate ligament", r"ligamento cruzado anterior", r"\blca\b"],
        pathology=[r"\btear", r"\btorn\b", r"ruptur", r"sprain", r"rotur", r"desgarr", r"esguinc", r"discontinuit"],
    ),
    "mcl_injury": dict(
        anatomy=[r"\bmcl\b", r"medial collateral ligament", r"ligamento colateral medial", r"ligamento lateral interno"],
        pathology=[r"\btear", r"\btorn\b", r"ruptur", r"sprain", r"rotur", r"desgarr", r"esguinc"],
    ),
    "medial_meniscus_tear": dict(
        anatomy=[r"medial meniscus", r"menisco medial"],
        pathology=[r"\btear", r"\btorn\b", r"rotur", r"desgarr", r"extrusion", r"extrusi[o0]n", r"macerat"],
    ),
    "lateral_meniscus_tear": dict(
        anatomy=[r"lateral meniscus", r"menisco lateral"],
        pathology=[r"\btear", r"\btorn\b", r"rotur", r"desgarr", r"extrusion", r"extrusi[o0]n", r"macerat"],
    ),
    "oa_medial_compartment": dict(
        anatomy=[r"medial compartment", r"medial femorotibial", r"medial femoral condyle",
                 r"medial tibial plateau", r"compartimento medial"],
        pathology=_OA_PATHOLOGY_CUES,
    ),
    "oa_lateral_compartment": dict(
        anatomy=[r"lateral compartment", r"lateral femorotibial", r"lateral femoral condyle",
                 r"lateral tibial plateau", r"compartimento lateral"],
        pathology=_OA_PATHOLOGY_CUES,
    ),
    "oa_patellofemoral_compartment": dict(
        anatomy=[r"patellofemoral", r"femoropatelar", r"femororrotulian", r"patelofemoral"],
        pathology=_OA_PATHOLOGY_CUES,
    ),
    "effusion": dict(
        anatomy=[r"\beffusion\b", r"derrame articular", r"\bderrame\b", r"efusi[o0]n"],
        pathology=[r"\beffusion\b", r"derrame", r"efusi[o0]n", r"fluid collection", r"joint fluid"],
    ),
    "synovitis": dict(
        anatomy=[r"synovit", r"sinovit", r"synovial (thickening|hypertrophy|proliferation)", r"engrosamiento sinovial"],
        pathology=[r"synovit", r"sinovit", r"synovial (thickening|hypertrophy|proliferation)", r"engrosamiento sinovial"],
    ),
    "bakers_cyst": dict(
        anatomy=[r"baker'?s? cyst", r"popliteal cyst", r"quiste de baker", r"quiste poplite"],
        pathology=[r"baker'?s? cyst", r"popliteal cyst", r"quiste de baker", r"quiste poplite"],
    ),
    "bone_contusion": dict(
        anatomy=[r"bone (marrow )?contusion", r"bone (marrow )?edema", r"contusi[o0]n [o0]sea", r"edema [o0]seo", r"edema medular"],
        pathology=[r"bone (marrow )?contusion", r"bone (marrow )?edema", r"contusi[o0]n [o0]sea", r"edema [o0]seo", r"edema medular"],
    ),
    "fracture": dict(
        anatomy=[r"\bfractur"],
        pathology=[r"\bfractur", r"cortical (break|disruption)", r"trabecular fracture"],
    ),
}

# Negation/normality cues, pooled across English and Spanish (matches the
# no-language-routing rule above — a report is never pre-classified by
# language before cues are tested). Co-occurrence within the same clause
# as an anatomy cue is what scopes a negation to that finding; this is a
# clause-level check, not a directional one (the reviewed prvsiyan
# notebook's "directional negation" refinement — scoping by which side of
# the anatomy mention the negation falls on — is a candidate improvement
# if a future validation run shows clause-level scoping misfiring on
# multi-finding sentences, not implemented here).
_NEGATION_CUES = [
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\babsent\b", r"\bnormal\b",
    r"\bintact\b", r"\bunremarkable\b", r"within normal limit",
    r"no evidence of", r"no sign", r"negative for",
    r"\bsin\b", r"\bausen", r"\bninguna\b", r"\bnegativ",
    r"dentro de (los )?l[i0]mites normales", r"no se (observa|evidencia|aprecia)",
]
_NEGATION_RE = re.compile("|".join(_NEGATION_CUES))

_COMPILED_LEXICON = {
    finding: (re.compile("|".join(cues["anatomy"])), re.compile("|".join(cues["pathology"])))
    for finding, cues in FINDING_LEXICON.items()
}


def label_report(report_text: str, finding: str) -> float:
    """Return a soft label in [0, 1] for one finding from one report.

    Per-clause rule: a clause counts as evidence for `finding` only if it
    matches that finding's `anatomy` cue (i.e. it's actually talking
    about the relevant structure). Within such a clause, a negation cue
    (`_NEGATION_RE`) wins over a pathology cue if both are present —
    validated as the right priority in notebooks/03_labeler_validation.ipynb
    section D-E: the opposite priority (pathology wins) scored effusion
    at AUC 0.438, *below* random, because "no effusion" contains the word
    "effusion" itself (anatomy == pathology cue list for entity-only
    findings), so a naive "pathology mention present -> positive" rule
    voted every negated mention as positive.

    A study with no clause matching `finding`'s anatomy cue at all
    returns 0.5 (abstain), not 0.0 — per Data Programming (Ratner et al.
    2016)/Snorkel, a labeling function's failure mode should be silence,
    not a confident wrong answer, since silence is what a caller
    combining several labeling functions can choose to down-weight. If
    any matching clause votes positive, the report is labeled positive
    (an assertion anywhere outweighs normal findings elsewhere, e.g. a
    multi-compartment report normal everywhere except one torn
    meniscus). Otherwise, if any matching clause votes negative, the
    report is labeled negative.
    """
    anatomy_re, pathology_re = _COMPILED_LEXICON[finding]
    votes = []
    for clause in _clauses(report_text):
        if not anatomy_re.search(clause):
            continue
        if _NEGATION_RE.search(clause):
            votes.append(0.0)
        elif pathology_re.search(clause):
            votes.append(1.0)
        # else: anatomy mentioned with neither cue -> ambiguous, no vote.
    if not votes:
        return 0.5
    return 1.0 if max(votes) == 1.0 else 0.0


def label_reports(reports: pd.DataFrame, findings: "list[str]") -> pd.DataFrame:
    """Apply label_report() across all studies and findings.

    `reports` must have a `Report` column and a `StudyInstanceUID`
    column (or be indexed by it already). Returns one soft-label column
    per finding, indexed by `StudyInstanceUID`.
    """
    if "StudyInstanceUID" in reports.columns:
        reports = reports.set_index("StudyInstanceUID")
    return pd.DataFrame({
        finding: reports["Report"].apply(lambda text: label_report(text, finding))
        for finding in findings
    }, index=reports.index)


def report_group_key(report_text: str) -> str:
    """Return a stable hash of the report text, for CV grouping.

    Both reference notebooks flag "shared reports" as a validation leak:
    some reports are byte-identical templates across studies (e.g. a
    template used for an unremarkable knee), so every study sharing that
    template gets the same derived target vector. Splitting such a group
    across train/validation scores the model on a target whose source it
    was trained on. Group folds by this hash, not only by study/scanner
    id — see src/evaluate.py and plan Fase 5.

    Confirmed as a real (not just hypothetical) concern by
    notebooks/02_eda_reports.ipynb section E (2026-08-17): 54 duplicate
    groups across 206 of 4,407 studies (4.7%), and at least one template
    shared between a gold study and a weak (report-only) study — so this
    grouping protects gold-vs-weak splits too, not only weak-vs-weak.
    """
    raise NotImplementedError("Fill in once load_reports() is available.")
