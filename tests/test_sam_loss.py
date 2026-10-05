# -*- coding: utf-8 -*-
"""tests/test_sam_loss.py - Plan Phase 33/15.

Run with: pytest tests/test_sam_loss.py -v
"""
import math

import torch

from training.gan import SAMLoss


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def test_zero_vector():
    loss = SAMLoss()
    x = torch.zeros(2, 4, 3, 3, requires_grad=True)
    y = torch.zeros(2, 4, 3, 3)
    out = loss(x, y)
    assert _finite(out)
    out.backward()
    assert _finite(x.grad)


def test_identical_vectors():
    loss = SAMLoss()
    x = torch.randn(2, 4, 3, 3, requires_grad=True)
    out = loss(x, x.detach().clone())
    assert _finite(out)
    # near-zero angle for identical spectra - bounded by the angle_eps
    # clamp trade-off documented in SAMLoss (default angle_eps=1e-3 caps
    # the achievable minimum angle at ~sqrt(2*angle_eps) =~ 0.045 rad).
    assert out.item() < 0.05
    out.backward()
    assert _finite(x.grad)


def test_near_identical_vectors():
    loss = SAMLoss()
    x = torch.randn(2, 4, 3, 3, requires_grad=True)
    y = x.detach().clone() + 1e-6 * torch.randn_like(x)
    out = loss(x, y)
    assert _finite(out)
    out.backward()
    assert _finite(x.grad)


def test_orthogonal_vectors():
    loss = SAMLoss()
    x = torch.zeros(1, 2, 1, 1, requires_grad=True)
    with torch.no_grad():
        x[0, 0, 0, 0] = 1.0
    y = torch.zeros(1, 2, 1, 1)
    y[0, 1, 0, 0] = 1.0
    out = loss(x, y)
    assert _finite(out)
    assert abs(out.item() - math.pi / 2) < 1e-2
    out.backward()
    assert _finite(x.grad)


def test_large_magnitude():
    loss = SAMLoss()
    x = (torch.randn(2, 8, 4, 4) * 1e6).requires_grad_(True)
    y = torch.randn(2, 8, 4, 4) * 1e6
    out = loss(x, y)
    assert _finite(out)
    out.backward()
    assert _finite(x.grad)


def test_small_magnitude():
    loss = SAMLoss()
    x = (torch.randn(2, 8, 4, 4) * 1e-6).requires_grad_(True)
    y = torch.randn(2, 8, 4, 4) * 1e-6
    out = loss(x, y)
    assert _finite(out)
    out.backward()
    assert _finite(x.grad)


def test_mixed_precision():
    loss = SAMLoss()
    x = torch.randn(2, 8, 4, 4, dtype=torch.float16, requires_grad=True)
    y = torch.randn(2, 8, 4, 4, dtype=torch.float16)
    out = loss(x, y)
    assert out.dtype == torch.float32  # internal computation always FP32 (Phase 15)
    assert _finite(out)
    out.backward()
    assert _finite(x.grad)


def test_never_returns_nan_or_inf_batch():
    """Bulk fuzz check across a mix of pathological cases at once."""
    loss = SAMLoss()
    torch.manual_seed(0)
    for _ in range(200):
        x = torch.randn(4, 16, 5, 5) * torch.randint(0, 2, (1,)).item() * 1e3
        y = torch.randn(4, 16, 5, 5) * torch.randint(0, 2, (1,)).item() * 1e3
        x.requires_grad_(True)
        out = loss(x, y)
        assert _finite(out), "SAMLoss produced a non-finite value"
        out.backward()
        assert _finite(x.grad), "SAMLoss produced a non-finite gradient"
        x.grad = None
