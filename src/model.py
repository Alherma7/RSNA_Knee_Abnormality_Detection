"""Model builders: per-plane backbone + attention pooling over slots.

Source: community baselines reviewed for this competition (see
RESOURCES.md, "Comparable projects") — a study is represented as up to
six plane/sequence "slots", each slot's slices encoded by a shared
backbone (EfficientNet-B0/B3 or a small ViT such as DINOv2-small).
Hands-On Machine Learning, ch. 13, and Dive into Deep Learning, ch. 8,
cover the CNN backbone fundamentals this builds on.

Two design choices the reviewed notebooks treat as decisive, not
optional:
- Pool slots with per-finding attention, not a plain mean. Each finding
  is diagnostically read on particular sequences/planes; a uniform mean
  over slots dilutes the one slot carrying the evidence with several
  that do not.
- Fine-tune the encoder rather than freezing it. A frozen self-
  supervised natural-image encoder (e.g. DINOv2 out of the box) is
  bounded by a feature vocabulary learned from natural images, which
  does not resemble the signal in an MRI slice — every other axis
  (resolution, model size, slice coverage) runs into that same ceiling.
"""


def build_backbone(name: str, pretrained: bool = True):
    """Return an image backbone (e.g. EfficientNet-B0/B3, DINOv2-small)."""
    raise NotImplementedError("Fill in once a DL framework is chosen.")


def build_multiplane_model(backbone_name: str, n_findings: int):
    """Wire backbone + per-finding attention pooling over plane/sequence slots.

    Each finding o gets its own learned query q_o that attends over the
    slot embeddings (with a presence mask for studies missing a slot),
    rather than a shared mean pool — see module docstring.
    """
    raise NotImplementedError("Fill in once build_backbone() is validated.")


def predict(model, study_features) -> "dict[str, float]":
    """Score one study: per-finding probability, for inference/submission.

    The competition metric (macro ROC-AUC) is invariant to any strictly
    increasing transform of a label's scores, so this function should
    NOT calibrate or threshold — raw ranked scores are what the metric
    rewards. See src/evaluate.py for how multiple models get combined
    (percentile-rank blend, not raw-probability averaging).
    """
    raise NotImplementedError("Fill in once build_multiplane_model() works.")
