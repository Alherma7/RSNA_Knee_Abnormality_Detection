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

masked_finding_attention() is A2's realization of the first bullet,
graduated 2026-08-26 — see
docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md
for the full design (source: DINOv2-small backbone, 6 named slots from
src/data.py::load_slot_cache_shard). build_backbone()/
build_multiplane_model() below stay NotImplementedError until
notebooks/05v2_slot_attention_baseline.ipynb (which duplicates this
function inline, since Kaggle can't import this repo) produces real,
user-reviewed output — that's the part of A2 real data has to validate,
not something a synthetic test can substitute for.
"""

import torch


def masked_finding_attention(embeddings: "torch.Tensor", mask: "torch.Tensor",
                              query: "torch.Tensor", head_weight: "torch.Tensor",
                              head_bias: "torch.Tensor") -> "torch.Tensor":
    """Per-finding masked-softmax attention over slot embeddings, then a
    per-finding linear head — A2's pooling mechanism, independent of
    whatever backbone produced `embeddings`.

    embeddings: (B, S, D) - S slot embeddings per study.
    mask: (B, S) - 1.0 where a slot is present, 0.0 where absent. A
    masked-out slot gets attention weight 0 for every finding, verified
    in tests/test_model.py by perturbing a masked slot's values and
    confirming the output doesn't change.
    query: (O, D) - one learned attention query per finding.
    head_weight: (O, D), head_bias: (O,) - one linear D->1 head per
    finding, applied via the O rows of an nn.Linear(D, O)'s weight
    matrix (row o IS finding o's D->1 head) - avoids an O-way ModuleList
    of separate nn.Linear(D, 1) layers for the same result.

    Raises ValueError if any row of `mask` is all-zero: a masked softmax
    over an all -inf row is NaN by construction. Real corpus data has a
    measured minimum of 3 present slots per study (checked against
    train_series.csv, see project memory), so this should never fire in
    practice - it exists as an explicit failure mode instead of a silent
    NaN if that assumption is ever wrong.

    Source: docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md,
    section 3 (A2 v1 design, user-approved 2026-08-26).
    """
    if not (mask.sum(dim=1) > 0).all():
        raise ValueError("masked_finding_attention: a row has 0 present slots")

    scores = torch.einsum("od,bsd->bos", query, embeddings) / (embeddings.shape[-1] ** 0.5)
    expanded_mask = mask.unsqueeze(1).expand(-1, query.shape[0], -1)
    scores = scores.masked_fill(expanded_mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)

    context = torch.einsum("bos,bsd->bod", weights, embeddings)
    logits = (context * head_weight.unsqueeze(0)).sum(-1) + head_bias

    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise RuntimeError("masked_finding_attention produced NaN/Inf logits")

    return logits


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
