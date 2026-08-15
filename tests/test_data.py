"""Unit tests for src/data.py.

Not yet written: every function in src/data.py currently raises
NotImplementedError until real DICOM/report files are downloaded and
validated in notebooks/01_eda_dicom.ipynb and 02_eda_reports.ipynb (see
README.md Next steps). Add one test per function here as each graduates
from the notebook, per structuring-ml-projects step 5 — in particular,
test_load_dicom_series should assert slices come back ordered by
physical position, not filename (see src/data.py's docstring on why
filename order is wrong for this data).
"""

import pytest

pytest.skip(
    "src/data.py functions are not implemented yet — see module docstring.",
    allow_module_level=True,
)
