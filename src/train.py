"""End-to-end pipeline: load -> label -> features -> fit -> evaluate -> output.

Wires together the functions in src/data.py, src/labelers.py,
src/features.py, src/model.py, and src/evaluate.py once each has been
validated in a notebook per the project plan. Nothing here should be the
first place a technique is tried — see notebooks/ and RESOURCES.md.
"""

from src import config


def run() -> None:
    """Run the full pipeline and write outputs/submission.csv."""
    raise NotImplementedError(
        "Wire this up phase by phase, once each src/ module's functions "
        "are validated in the notebooks — see README.md Next steps."
    )


if __name__ == "__main__":
    run()
