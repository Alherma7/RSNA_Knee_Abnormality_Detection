"""Unit tests for src/model.py.

masked_finding_attention() is graduated (2026-08-26, A2) — pure tensor
math, independent of the real DINOv2 backbone, verified here on small
hand-controlled embeddings. build_backbone()/build_multiplane_model()
stay NotImplementedError until the Kaggle validation notebook
(notebooks/05v2_slot_attention_baseline.ipynb) produces real, reviewed
output — see docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md
section 6. No tests for those yet.
"""

import pytest
import torch

from src.model import masked_finding_attention


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
