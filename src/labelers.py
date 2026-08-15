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

Validate against the gold subset (src.evaluate) before this module's
output is used to train the image model — see plan Fase 3.
"""

import pandas as pd


def label_report(report_text: str, finding: str) -> float:
    """Return a soft label in [0, 1] for one finding from one report.

    Placeholder: start with a small set of keyword/cue-based labeling
    functions per finding (see Building Machine Learning Systems, ch. 6,
    for the text-cleaning baseline). Unwrap hard-wrapped line breaks
    before sentence splitting — both reference notebooks note reports
    arrive hard-wrapped at a fixed column, so naive newline splitting
    breaks sentences without punctuation at the break. Do not wire this
    into training until it beats a trivial baseline on the gold subset.
    """
    raise NotImplementedError("Implement labeling functions per finding.")


def label_reports(reports: pd.DataFrame, findings: "list[str]") -> pd.DataFrame:
    """Apply label_report() across all studies and findings.

    Returns one soft-label column per finding, indexed by study id.
    """
    raise NotImplementedError("Fill in once label_report() is validated.")


def report_group_key(report_text: str) -> str:
    """Return a stable hash of the report text, for CV grouping.

    Both reference notebooks flag "shared reports" as a validation leak:
    some reports are byte-identical templates across studies (e.g. a
    template used for an unremarkable knee), so every study sharing that
    template gets the same derived target vector. Splitting such a group
    across train/validation scores the model on a target it was trained
    on. Group folds by this hash, not only by study/scanner id — see
    src/evaluate.py and plan Fase 5.
    """
    raise NotImplementedError("Fill in once load_reports() is available.")
