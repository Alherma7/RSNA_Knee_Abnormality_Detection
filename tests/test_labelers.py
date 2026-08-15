"""Unit tests for src/labelers.py.

Not yet written: label_report() is not implemented yet — validate the
labeling functions against the gold subset in
notebooks/03_labeler_validation.ipynb first (see README.md Next steps).
Once individual labeling functions exist, test known edge cases here:
an empty report, a report with no cue for a finding (should abstain, not
emit a confident negative — see src/labelers.py docstring), and a
hard-wrapped report where a sentence breaks across two lines.
"""

import pytest

pytest.skip(
    "src/labelers.py functions are not implemented yet — see module docstring.",
    allow_module_level=True,
)
