"""Unit tests for src/labelers.py.

Covers the plumbing (normalize/unwrap/clause-splitting, cue matching,
abstention), not model quality — model quality is what
notebooks/03_labeler_validation.ipynb's macro-ROC-AUC gate is for
(structuring-ml-projects step 5). Several tests are direct regressions
for real bugs caught during that validation and a later audit
(2026-08-18): negation priority within a sub-clause (effusion scored
below random until fixed), full-clause negation scoping being too wide
(a negation for one finding vetoing an unrelated finding's assertion in
the same sentence), the bullet-marker unwrap bug (caught in
notebooks/02_eda_reports.ipynb), and the sentence splitter requiring
whitespace after terminal punctuation.
"""

import pandas as pd

from src.labelers import _unwrap, label_report, label_reports, report_group_key


def test_label_report_abstains_on_empty_report():
    assert label_report("", "acl_injury") == 0.5


def test_label_report_abstains_on_missing_report():
    assert label_report(None, "acl_injury") == 0.5


def test_label_report_abstains_when_finding_not_mentioned():
    # Only meniscus is discussed; nothing here should trigger the ACL cues.
    text = "Lateral meniscus tear noted. All other structures intact."
    assert label_report(text, "acl_injury") == 0.5


def test_label_report_detects_positive_assertion():
    text = "There is a complete tear of the ACL with retraction."
    assert label_report(text, "acl_injury") == 1.0


def test_label_report_detects_explicit_negation():
    text = "The ACL is intact. No other abnormality."
    assert label_report(text, "acl_injury") == 0.0


def test_label_report_negation_wins_over_pathology_in_same_clause():
    # Regression test: the entity name IS the pathology cue for effusion,
    # so "no effusion" matches both the anatomy cue and the pathology cue
    # in the same clause. An earlier version let pathology win in that
    # case, which scored effusion at ROC-AUC 0.438 (below random) in
    # notebooks/03_labeler_validation.ipynb section D — every negated
    # mention was voted positive. Negation must win.
    assert label_report("No joint effusion is seen.", "effusion") == 0.0
    assert label_report("Small joint effusion is present.", "effusion") == 1.0


def test_label_report_negation_does_not_leak_across_findings_in_same_clause():
    # Regression test (audit, 2026-08-18): a full-clause-level negation
    # check let "without effusion" veto the meniscus tear earlier in the
    # SAME clause, because both cues lived in one comma-free sentence.
    # "without" is pre-nominal (negates what follows it), so the fix
    # scopes it to a new sub-clause that starts right before it.
    text = "Complex tear of the medial meniscus posterior horn without effusion."
    assert label_report(text, "medial_meniscus_tear") == 1.0
    assert label_report(text, "effusion") == 0.0


def test_label_report_negation_scoping_handles_but():
    text = "The ACL is torn but the PCL is intact."
    assert label_report(text, "acl_injury") == 1.0


def test_label_report_coordinated_negation_list_still_applies_to_every_item():
    # Regression test for a real failure of an earlier version of this
    # fix: splitting on every comma (to handle the "but" case above)
    # broke coordinated lists sharing one negation, e.g. "no X, Y, or Z"
    # — measured directly against the 58-row gold set as a macro ROC-AUC
    # drop from 0.688 to 0.674, reverted. A negation's argument is
    # deliberately left unbounded by commas (see _negation_applies()) so
    # it still reaches every item in a list like this one.
    text = "No effusion, synovitis, or bone contusion identified."
    assert label_report(text, "effusion") == 0.0
    assert label_report(text, "synovitis") == 0.0
    assert label_report(text, "bone_contusion") == 0.0


def test_label_report_known_limitation_comma_joined_independent_clauses():
    # Documents a known, accepted gap rather than hiding it: two fully
    # independent clauses joined only by a comma (no "but"/"pero", no
    # repeated anatomy word) are NOT reliably scoped apart, because
    # splitting on bare commas regresses the real coordinated-list case
    # above. This exact sentence was not confirmed present in the real
    # corpus (unlike the "but" and Spanish comma examples elsewhere in
    # this file, which are true excerpts from data/raw/train.csv) — left
    # as a documented gap rather than chased further, per this project's
    # explicit skepticism about over-fitting the lexicon to hand-picked
    # examples (see the module docstring).
    text = "There is a radial tear of the medial meniscus, the lateral meniscus is normal."
    assert label_report(text, "medial_meniscus_tear") == 0.0  # ideally 1.0 -- known gap


def test_label_report_predicate_negation_still_wins_after_scoping_fix():
    # Sanity check that scoping to sub-clauses didn't break the simple
    # predicate case: "is intact" must stay attached to "ACL" (a
    # predicate cue must NOT start a new sub-clause, unlike a pre-nominal
    # cue like "without" above) — see _sub_clauses() docstring.
    assert label_report("The ACL is intact.", "acl_injury") == 0.0


def test_label_report_splits_sentences_with_no_space_after_period():
    # Regression test (audit, 2026-08-18): 677/4407 real reports (15.4%)
    # run findings together with no space after the period, e.g.
    # "acl tear.mcl normal.medial meniscus tear." — the old sentence
    # splitter required trailing whitespace and merged these into one
    # clause, which fed the negation-leak bug above.
    text = "ACL tear.MCL normal.Medial meniscus tear."
    assert label_report(text, "acl_injury") == 1.0
    assert label_report(text, "mcl_injury") == 0.0
    assert label_report(text, "medial_meniscus_tear") == 1.0


def test_label_report_rejoins_hard_wrapped_sentence_across_finding_words():
    # "anterior cruciate ligament" is split by a mid-sentence line break
    # with no closing punctuation before it — exactly the hard-wrap
    # pattern measured in notebooks/02_eda_reports.ipynb section C. If
    # unwrap() didn't rejoin the lines, neither line alone would contain
    # the full "anterior cruciate ligament" anatomy cue.
    text = "Complete tear of the anterior cruciate\nligament with retraction."
    assert label_report(text, "acl_injury") == 1.0


def test_unwrap_does_not_merge_a_new_bulleted_statement():
    # Regression test for the bug found in notebooks/02_eda_reports.ipynb
    # section C: a bullet marker ">" is neither upper- nor lowercase, so
    # `s[:1].isupper()` was always False for a bulleted line, and every
    # bulleted line after a non-punctuation-ending line got merged into
    # it as if it were a wrapped continuation — even when the bulleted
    # line was a new, independent, capitalized statement. The fix strips
    # the bullet before checking capitalization, so this stays unmerged.
    text = "increase signal intensity, likely due to sprain\n> No definite tear of the meniscus"
    assert _unwrap(text) == text


def test_unwrap_still_merges_a_genuine_wrapped_continuation():
    # Sanity check that the bullet-marker fix didn't disable unwrapping
    # itself: a plain (non-bulleted) lowercase continuation still merges.
    text = "Complete tear of the anterior cruciate\nligament with retraction."
    assert _unwrap(text) == "Complete tear of the anterior cruciate ligament with retraction."


def test_report_group_key_is_stable_and_case_diacritic_insensitive():
    assert report_group_key("Rodilla normal.") == report_group_key("RODILLA NORMAL.")
    assert report_group_key("rótula normal") == report_group_key("rotula normal")


def test_report_group_key_ignores_line_wrap_position():
    # Regression test (2026-08-18): reusing _normalize() here (which
    # deliberately keeps newlines intact for _clauses()) under-counted
    # real duplicate templates that only differ by where the line wraps
    # — measured as 50 groups/192 studies instead of the real 54/206
    # against data/raw/train.csv. This function needs the opposite of
    # _normalize()'s newline handling.
    wrapped_a = "Complete tear of the anterior\ncruciate ligament."
    wrapped_b = "Complete tear of the anterior cruciate\nligament."
    assert report_group_key(wrapped_a) == report_group_key(wrapped_b)


def test_report_group_key_differs_for_different_text():
    assert report_group_key("ACL torn.") != report_group_key("ACL intact.")


def test_report_group_key_handles_missing_report():
    assert report_group_key(None) == report_group_key("")
    assert report_group_key(float("nan")) == report_group_key("")


def test_label_reports_returns_one_column_per_finding_indexed_by_study():
    reports = pd.DataFrame({
        "StudyInstanceUID": ["s1", "s2"],
        "Report": [
            "There is a complete tear of the ACL.",
            "The ACL is intact. No other abnormality.",
        ],
    })
    result = label_reports(reports, ["acl_injury", "fracture"])
    assert list(result.columns) == ["acl_injury", "fracture"]
    assert list(result.index) == ["s1", "s2"]
    assert result.loc["s1", "acl_injury"] == 1.0
    assert result.loc["s2", "acl_injury"] == 0.0
    assert result.loc["s1", "fracture"] == 0.5  # not mentioned -> abstain
