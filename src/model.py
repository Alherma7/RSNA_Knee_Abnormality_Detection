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

masked_finding_attention()/build_backbone()/build_multiplane_model() are
A2's realization of the first bullet — see
docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md for
the full design. Graduated 2026-08-26 from
notebooks/05v2_slot_attention_baseline.ipynb (which carries its own
duplicated copy of this same logic, since Kaggle can't import this
repo), after that notebook produced real, user-reviewed output on
Kaggle: 0.7689 gold macro-AUC, fold 0, real DINOv2-small weights, real
A3 cache, 12 epochs. `predict()` stays NotImplementedError — that needs
the test-time cache/inference pipeline, not built yet.
"""

import timm
import torch
import torch.nn as nn


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


def build_backbone(name: str = "vit_small_patch14_dinov2.lvd142m",
                    pretrained: bool = True, img_size: int = 224):
    """Return a timm image backbone, feature-extraction mode (no head).

    Default is DINOv2-small — the exact tagged identifier confirmed
    2026-08-26 (the bare name without `.lvd142m` does not resolve, see
    RESOURCES.md). Native pretrained resolution is 518x518; `img_size`
    interpolates the position embeddings down to our A3 cache's 224x224
    — confirmed working on real Kaggle data 2026-08-26 (A2 v1's run).
    `num_classes=0` drops the classification head, returning pooled
    features of dimension `model.num_features` (384 for DINOv2-small).
    """
    return timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=img_size)


class SlotAttentionModel(nn.Module):
    """DINOv2-small backbone (shared, fine-tuned) + per-finding masked
    attention over up to 6 slots + one small linear head per finding.

    See docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md
    section 3 for the full design rationale, and masked_finding_attention()
    above for the pooling math this wraps.
    """

    def __init__(self, n_findings: int, n_slots: int,
                 backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
                 pretrained: bool = True, unfreeze_last: int = 6):
        super().__init__()
        self.backbone = build_backbone(backbone_name, pretrained=pretrained)
        embed_dim = self.backbone.num_features

        for p in self.backbone.parameters():
            p.requires_grad = False
        for block in self.backbone.blocks[-unfreeze_last:]:
            for p in block.parameters():
                p.requires_grad = True

        self.query = nn.Parameter(torch.randn(n_findings, embed_dim) * (embed_dim ** -0.5))
        self.heads = nn.Linear(embed_dim, n_findings)
        self.embed_dim = embed_dim
        self.n_findings = n_findings
        self.n_slots = n_slots

    def forward(self, slot_images: "torch.Tensor", slot_mask: "torch.Tensor") -> "torch.Tensor":
        """slot_images: (B, n_slots, 3, H, W). slot_mask: (B, n_slots). Returns (B, n_findings) logits."""
        B, S, C, H, W = slot_images.shape
        if S != self.n_slots or C != 3:
            raise ValueError(
                f"expected slot_images (*, {self.n_slots}, 3, H, W), got {tuple(slot_images.shape)}"
            )
        if tuple(slot_mask.shape) != (B, S):
            raise ValueError(f"expected slot_mask ({B}, {S}), got {tuple(slot_mask.shape)}")

        flat = slot_images.view(B * S, C, H, W)
        embeddings = self.backbone(flat).view(B, S, self.embed_dim)
        return masked_finding_attention(
            embeddings, slot_mask, self.query, self.heads.weight, self.heads.bias
        )


def build_multiplane_model(backbone_name: str, n_findings: int, n_slots: int = 6,
                            pretrained: bool = True, unfreeze_last: int = 6) -> SlotAttentionModel:
    """Wire backbone + per-finding masked attention pooling over slots.

    Each finding o gets its own learned query q_o that attends over the
    slot embeddings (masked for studies missing a slot), rather than a
    shared mean pool — see module docstring and masked_finding_attention().

    Graduated 2026-08-26 from
    notebooks/05v2_slot_attention_baseline.ipynb (A2), where this exact
    construction (n_slots=6, unfreeze_last=6, DINOv2-small) trained for
    real: 0.7689 gold macro-AUC, fold 0.
    """
    return SlotAttentionModel(
        n_findings=n_findings, n_slots=n_slots, backbone_name=backbone_name,
        pretrained=pretrained, unfreeze_last=unfreeze_last,
    )


def predict(model, study_features) -> "dict[str, float]":
    """Score one study: per-finding probability, for inference/submission.

    The competition metric (macro ROC-AUC) is invariant to any strictly
    increasing transform of a label's scores, so this function should
    NOT calibrate or threshold — raw ranked scores are what the metric
    rewards. See src/evaluate.py for how multiple models get combined
    (percentile-rank blend, not raw-probability averaging).
    """
    raise NotImplementedError("Fill in once build_multiplane_model() works.")
