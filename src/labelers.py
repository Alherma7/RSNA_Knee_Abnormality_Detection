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

import hashlib
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


_SENT_SPLIT_RE = re.compile(r"(?<=[.;!?])\s*|\n+")


def _clauses(text: str) -> "list[str]":
    """Normalize, unwrap, then split into sentence/line-level clauses.

    Audit finding (2026-08-18): the previous pattern required whitespace
    after `.;!?` to split, so reports with no space after a period (e.g.
    "acl tear.mcl normal.medial meniscus tear.") — 15.4% of the 4,407
    reports (677), measured directly — collapsed multiple findings into
    one clause, which then fed the negation-scoping bug below (a
    negation for one finding could veto an unrelated finding's
    assertion in the same run-together "clause"). Splitting
    unconditionally after `.;!?` (whitespace or not) fixes this; the
    empty pieces this can produce are already filtered out below.
    """
    norm = _normalize(_unwrap(text))
    return [c.strip() for c in _SENT_SPLIT_RE.split(norm) if c and c.strip()]


def _sub_clauses(clause: str) -> "list[str]":
    """Split a clause at coordinating conjunctions a negation can't cross.

    Audit finding (2026-08-18): `label_report()` used to let any
    negation cue anywhere in a clause veto a pathology assertion
    anywhere else in that same clause — wrong for compound sentences
    like "The ACL is torn but the PCL is intact", where "intact" negates
    the PCL, not the ACL. `but`/`pero`/`aunque`/`although` reliably
    introduce a genuinely separate assertion in this corpus (checked
    against the real gold reports below), so splitting there is safe.

    A comma is NOT a safe split point here, unlike those conjunctions:
    an earlier version of this fix also split on `,`, which broke a
    pattern that is common in this corpus in both languages — naming a
    structure once and continuing to describe it across a comma without
    repeating the name (e.g. "menisco medial ... conservada, sin signos
    de rotura" or "no effusion, synovitis, or bone contusion
    identified"). Splitting there stranded the anatomy mention in one
    piece and the negation/pathology in another with no anatomy cue of
    its own, turning correct votes into wrong abstentions — measured
    directly: macro ROC-AUC on the 58-row gold set dropped from 0.688 to
    0.674 with comma-splitting in place, a real regression, reverted.
    Cross-entity negation scoping for cases that need finer-than-clause
    granularity (like the effusion example in `label_report()`'s
    docstring) is handled by `_negation_applies()` instead, which checks
    what a negation cue's local argument actually names rather than
    guessing from punctuation alone.
    """
    return [s.strip() for s in _SUB_CLAUSE_SPLIT_RE.split(clause) if s and s.strip()]


def _negation_applies(sub_clause: str, anatomy_re: "re.Pattern", pathology_re: "re.Pattern") -> bool:
    """Whether some negation cue in `sub_clause` actually negates `finding`.

    A negation cue only counts against `finding` if the text it plausibly
    governs — everything after a pre-nominal cue ("no", "without", "sin",
    ...) or everything before a predicate cue ("... is normal/intact",
    ...) — itself names `finding`'s own anatomy or pathology. This is
    what lets "tear of the medial meniscus without effusion" correctly
    NOT negate the meniscus tear (the argument of "without" is
    "effusion", naming a different finding entirely) while still
    correctly negating "menisco medial ... conservada, sin signos de
    rotura" (the argument of "sin" names this finding's own pathology
    word, "rotura", even without repeating the anatomy word) — see
    `_sub_clauses()` for why this replaces punctuation-based splitting
    for this specific ambiguity instead of comma-splitting.
    """
    for m in _NEGATION_SCOPE_RE.finditer(sub_clause):
        arg = sub_clause[m.end():]
        if anatomy_re.search(arg) or pathology_re.search(arg):
            return True
    for m in _NEGATION_PREDICATE_RE.finditer(sub_clause):
        arg = sub_clause[:m.start()]
        if anatomy_re.search(arg) or pathology_re.search(arg):
            return True
    return False


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
# language before cues are tested). Split into two groups by where the
# cue sits relative to what it negates (see _sub_clauses() docstring):
# pre-nominal cues precede the negated noun ("no effusion") and start a
# new sub-clause scope; predicate cues follow the subject they negate
# ("ACL is intact") and must NOT start a new scope, or they'd get split
# away from the anatomy mention they belong to.
_NEGATION_SCOPE_CUES = [
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\babsent\b",
    r"no evidence of", r"no sign", r"negative for",
    r"\bsin\b", r"\bausen", r"\bninguna\b", r"\bnegativ",
    r"no se (?:observa|evidencia|aprecia)",
]
_NEGATION_PREDICATE_CUES = [
    r"\bnormal\b", r"\bintact\b", r"\bunremarkable\b", r"within normal limit",
    r"dentro de (los )?l[i0]mites normales",
]
_NEGATION_CUES = _NEGATION_SCOPE_CUES + _NEGATION_PREDICATE_CUES
_NEGATION_RE = re.compile("|".join(_NEGATION_CUES))
_NEGATION_SCOPE_RE = re.compile("|".join(_NEGATION_SCOPE_CUES))
_NEGATION_PREDICATE_RE = re.compile("|".join(_NEGATION_PREDICATE_CUES))
_SUB_CLAUSE_SPLIT_RE = re.compile(r"\s+but\s+|\s+pero\s+|\s+aunque\s+|\s+although\s+")

_COMPILED_LEXICON = {
    finding: (re.compile("|".join(cues["anatomy"])), re.compile("|".join(cues["pathology"])))
    for finding, cues in FINDING_LEXICON.items()
}


def label_report(report_text: str, finding: str) -> float:
    """Return a soft label in [0, 1] for one finding from one report.

    Per-sub-clause rule: a sub-clause (see `_sub_clauses()` — a clause
    split further at `but`/`pero`/`aunque`/`although`) counts as evidence
    for `finding` only if it matches that finding's `anatomy` cue (i.e.
    it's actually talking about the relevant structure). Within such a
    sub-clause, a negation cue wins over a pathology cue if the negation
    actually applies to `finding` (see `_negation_applies()`) — negation
    winning over pathology when both are present was validated as the
    right priority in notebooks/03_labeler_validation.ipynb section D-E:
    the opposite priority (pathology wins) scored effusion at AUC 0.438,
    *below* random, because "no effusion" contains the word "effusion"
    itself (anatomy == pathology cue list for entity-only findings), so
    a naive "pathology mention present -> positive" rule voted every
    negated mention as positive.

    A study with no sub-clause matching `finding`'s anatomy cue at all
    returns 0.5 (abstain), not 0.0 — per Data Programming (Ratner et al.
    2016)/Snorkel, a labeling function's failure mode should be silence,
    not a confident wrong answer, since silence is what a caller
    combining several labeling functions can choose to down-weight. If
    any matching sub-clause votes positive, the report is labeled
    positive (an assertion anywhere outweighs normal findings elsewhere,
    e.g. a multi-compartment report normal everywhere except one torn
    meniscus). Otherwise, if any matching sub-clause votes negative, the
    report is labeled negative.
    """
    anatomy_re, pathology_re = _COMPILED_LEXICON[finding]
    votes = []
    for clause in _clauses(report_text):
        for sub in _sub_clauses(clause):
            if not anatomy_re.search(sub):
                continue
            if _negation_applies(sub, anatomy_re, pathology_re):
                votes.append(0.0)
            elif pathology_re.search(sub):
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

    Deliberately does NOT reuse `_normalize()`: that function collapses
    only spaces/tabs, not newlines, because `_clauses()` needs newlines
    intact as a clause boundary. This function needs the opposite —
    two studies sharing a template but wrapped at different line widths
    must hash the same — so it collapses ALL whitespace (spaces, tabs,
    newlines alike) to match the exact normalization
    notebooks/02_eda_reports.ipynb section E used to measure the 54
    duplicate groups above (case+NFKD-diacritic fold, then any
    whitespace run to a single space). Reusing `_normalize()` here
    instead was tried and measured to under-count: 50 groups / 192
    studies instead of 54 / 206, against the real `data/raw/train.csv`
    (2026-08-18) — the missing 4 groups/14 studies were templates that
    only differed by line-wrap position.

    A missing/non-string report still needs a key so every row can be
    grouped: this normalizes to `""` for non-string input, so every
    missing-report row lands in the same group as every other one —
    fine here since `Report` was confirmed non-null on all 4,407 rows in
    `train.csv` (Fase 1); `test.csv` has no `Report` column at all, but
    this function is never called at inference time (see module
    docstring).
    """
    if not isinstance(report_text, str):
        normalized = ""
    else:
        t = unicodedata.normalize("NFKD", report_text.lower())
        t = "".join(ch for ch in t if not unicodedata.combining(ch))
        normalized = re.sub(r"\s+", " ", t).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
