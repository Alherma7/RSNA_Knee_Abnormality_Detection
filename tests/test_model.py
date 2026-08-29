"""Unit tests for src/model.py.

masked_finding_attention() is graduated (2026-08-26, A2) — pure tensor
math, independent of the real DINOv2 backbone, verified here on small
hand-controlled embeddings. build_backbone()/build_multiplane_model()/
SlotAttentionModel are graduated too (2026-08-26), after
notebooks/05v2_slot_attention_baseline.ipynb produced real, reviewed
output on Kaggle (0.7689 gold macro-AUC, fold 0) — see
docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md
section 6. Tests here use `pretrained=False` (real architecture, random
weights) to stay hermetic/fast — no network download, no GPU needed;
the real-weights/real-data training result isn't re-proven here.
"""

import pytest
import torch

from src.model import build_backbone, build_multiplane_model, masked_finding_attention


def test_masked_finding_attention_ignores_masked_slot_values():
    torch.manual_seed(0)
    embeddings = torch.randn(1, 3, 4)  # (B=1, S=3, D=4)
    query = torch.randn(2, 4)          # (O=2, D=4)
    head_weight = torch.randn(2, 4)
    head_bias = torch.randn(2)
    mask = torch.tensor([[1.0, 1.0, 0.0]])  # slot 2 absent

    baseline = masked_finding_attention(embeddings, mask, query, head_weight, head_bias)

    perturbed = embeddings.clone()
    perturbed[0, 2] = torch.randn(4) * 1000  # wildly different values in the masked slot
    perturbed_out = masked_finding_attention(perturbed, mask, query, head_weight, head_bias)

    assert torch.allclose(baseline, perturbed_out, atol=1e-5)


def test_masked_finding_attention_uses_present_slots_normally():
    torch.manual_seed(1)
    embeddings = torch.randn(2, 2, 4)  # (B=2, S=2, D=4)
    query = torch.randn(1, 4)          # (O=1, D=4)
    head_weight = torch.randn(1, 4)
    head_bias = torch.randn(1)
    mask = torch.ones(2, 2)

    out = masked_finding_attention(embeddings, mask, query, head_weight, head_bias)

    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()


def test_masked_finding_attention_raises_on_a_fully_masked_row():
    embeddings = torch.randn(1, 2, 4)
    query = torch.randn(1, 4)
    head_weight = torch.randn(1, 4)
    head_bias = torch.randn(1)
    mask = torch.zeros(1, 2)

    with pytest.raises(ValueError):
        masked_finding_attention(embeddings, mask, query, head_weight, head_bias)


def test_build_backbone_returns_dinov2_small_with_expected_embed_dim():
    backbone = build_backbone(pretrained=False)
    assert backbone.num_features == 384


def test_build_multiplane_model_forward_produces_correct_shape():
    torch.manual_seed(0)
    model = build_multiplane_model("vit_small_patch14_dinov2.lvd142m", n_findings=12,
                                    n_slots=6, pretrained=False)
    images = torch.randn(2, 6, 3, 224, 224)
    mask = torch.ones(2, 6)

    logits = model(images, mask)

    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()


def test_build_multiplane_model_forward_raises_on_wrong_slot_count():
    model = build_multiplane_model("vit_small_patch14_dinov2.lvd142m", n_findings=12,
                                    n_slots=6, pretrained=False)
    images = torch.randn(2, 5, 3, 224, 224)  # 5 slots, not 6
    mask = torch.ones(2, 5)

    with pytest.raises(ValueError):
        model(images, mask)


def test_build_multiplane_model_forward_raises_on_mismatched_mask_shape():
    model = build_multiplane_model("vit_small_patch14_dinov2.lvd142m", n_findings=12,
                                    n_slots=6, pretrained=False)
    images = torch.randn(2, 6, 3, 224, 224)
    mask = torch.ones(3, 6)  # batch size doesn't match images

    with pytest.raises(ValueError):
        model(images, mask)


def test_build_multiplane_model_forward_with_18_pseudo_slots():
    # A2 v2 (graduated 2026-08-29): n_slots=18 (all 3 anchor groups on all
    # 6 slots, via src.features.expand_slot_groups()) needs zero model
    # code changes -- n_slots is a plain constructor parameter, verified
    # here the same way n_slots=6 already was. Real pooled 4-fold gold
    # macro-AUC 0.8009 (docs/superpowers/specs/2026-08-28-a2v2-multigroup-
    # slot-attention-design.md section 5.2).
    torch.manual_seed(0)
    model = build_multiplane_model("vit_small_patch14_dinov2.lvd142m", n_findings=12,
                                    n_slots=18, pretrained=False)
    images = torch.randn(2, 18, 3, 224, 224)
    mask = torch.ones(2, 18)

    logits = model(images, mask)

    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()
