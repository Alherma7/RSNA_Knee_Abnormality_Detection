"""Unit tests for src/features.py.

Not yet written: feature functions are not implemented yet — validate
against real DICOM data in notebooks/04_baseline_cnn.ipynb first (see
README.md Next steps). Once implemented, test normalize_laterality()
against a known left/right pair (flipped images should become
pixel-identical up to the flip) and build_25d_triplet() against a
boundary slice index (first/last slice, where a naive [n-gap, n+gap]
would go out of bounds).
"""

import pytest

pytest.skip(
    "src/features.py functions are not implemented yet — see module docstring.",
    allow_module_level=True,
)
