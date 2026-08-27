"""End-to-end pipeline: load -> label -> features -> fit -> evaluate -> output.

Wires together the functions in src/data.py, src/labelers.py,
src/features.py, src/model.py, and src/evaluate.py once each has been
validated in a notebook per the project plan. Nothing here should be the
first place a technique is tried — see notebooks/ and RESOURCES.md.

NOTE (A4, 2026-08-27): this project's Kaggle notebooks cannot `import
src` (confirmed 2026-08-26, see docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md
and .../2026-08-27-a4-submission-pipeline-design.md) — both real
training (notebooks/05v2_.../06v2_...) and the real submission pipeline
(notebooks/07v1_a2_submission_inference.ipynb) run as self-contained
Kaggle notebooks, never as this function. run() staying
NotImplementedError is not unfinished work to pick up later; it reflects
that a locally-runnable pipeline script was never going to be what
actually executes on Kaggle for this project.
"""

from src import config


def run() -> None:
    """Deliberately unimplemented — see the module docstring above for why."""
    raise NotImplementedError(
        "This project's real pipelines run as self-contained Kaggle "
        "notebooks (see notebooks/07v1_a2_submission_inference.ipynb for "
        "the real submission pipeline), not as this function — see the "
        "module docstring."
    )


if __name__ == "__main__":
    run()
