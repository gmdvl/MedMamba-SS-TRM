# -*- coding: utf-8 -*-
"""
test_latent_space_tuple_v18.py
================================
v18 F3. The two cases that produce today's empty `latent_space/` directory,
and the sampling flaw that made the two non-empty ones uninformative.

Run with: pytest tests/test_latent_space_tuple_v18.py -q
"""

import numpy as np
import pytest
import torch

from medmamba_ss_trm import MedMambaSSTRMConfig, MedMambaSSTRM
from training.latent_space import extract_penultimate_features as frozen_extract
from training.latent_space_v18 import (
    extract_penultimate_features, unwrap_for_features,
)

_SMALL = dict(recursive=True, dims=(32,), depths=(1,), d_state=8, d_ctx=32,
              patch_size=1, spectral_depth=1, trm_dim=32, trm_core_layers=1,
              trm_n_latent=2, trm_n_improve=2, trm_deep_supervision_steps=2,
              trm_mixer="mlp", trm_act_halting=False)

C, HW, N, BS = 8, 5, 12, 4


def _model():
    return MedMambaSSTRM(MedMambaSSTRMConfig(**_SMALL), task="classification", num_classes=3)


def _loader(n_items: int, batch: int, three_tuple: bool):
    """A plain iterable of batches - `extract_penultimate_features` only ever
    iterates, so a DataLoader would add startup cost and no coverage."""
    def gen():
        for _ in range(n_items // batch):
            x = torch.randn(batch, C, HW, HW)
            y = torch.randint(0, 3, (batch,))
            yield (x, y, torch.randn(batch, 2, C)) if three_tuple else (x, y)
    return gen()


class ReconWrapper(torch.nn.Module):
    """The shape `MedMambaSSTRMLatentReconWrapperV2` presents: a `base_model`, and
    a forward returning a 2-tuple."""

    def __init__(self, inner):
        super().__init__()
        self.base_model = inner

    def forward(self, x, wavelengths=None, sensor_range=None):
        logits = self.base_model(x, wavelengths=wavelengths, sensor_range=sensor_range)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits, torch.zeros_like(x)


# ----------------------------------------------------------------------------
# The defect, pinned
# ----------------------------------------------------------------------------

def test_frozen_extractor_cannot_unpack_a_three_tuple_batch():
    """`training/latent_space.py:62` is `for x, y in loader:`.

    This is why EVERY `--recon_mode latent` run has an empty `latent_space/`:
    the ValueError is swallowed as a warning at `trainerg_v12.py:1178`. If this
    stops raising, the frozen file has been changed.
    """
    with pytest.raises(ValueError):
        frozen_extract(_model(), _loader(N, BS, three_tuple=True), "cpu", max_samples=N)


# ----------------------------------------------------------------------------
# The fix
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("three_tuple", [False, True])
def test_tolerant_extractor_handles_both_batch_shapes(three_tuple):
    out = extract_penultimate_features(_model(), _loader(N, BS, three_tuple), "cpu",
                                        max_samples=N)
    assert out is not None, "extraction returned None - latent_space/ would be empty"
    features, labels = out
    assert features.ndim == 2 and features.shape[0] == labels.shape[0] == N
    assert np.isfinite(features).all()


def test_tolerant_extractor_handles_a_recon_wrapped_model():
    """The frozen version peels `_orig_mod` only, so a reconstruction-wrapped
    model falls through to the hook path - where `head(...)` is called with the
    backbone's feature DICT and `inputs[0]` is not a tensor."""
    out = extract_penultimate_features(ReconWrapper(_model()), _loader(N, BS, True), "cpu",
                                        max_samples=N)
    assert out is not None
    features, _ = out
    assert features.shape[0] == N


def test_unwrap_reaches_through_both_wrapper_kinds():
    inner = _model()

    class FakeCompiled(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self._orig_mod = m

    assert unwrap_for_features(inner) is inner
    assert unwrap_for_features(ReconWrapper(inner)) is inner
    assert unwrap_for_features(FakeCompiled(ReconWrapper(inner))) is inner


def test_wavelengths_are_forwarded_and_change_the_embedding():
    """`trainerg_v12.py:1173` passes no wavelengths, so the frozen path embeds
    with the index-encoding fallback (`medmamba_ss_trm.py:836`) rather than the
    encoding the model was trained under. The two must differ, or forwarding
    them would be pointless."""
    model = _model().eval()
    wl = np.linspace(400.0, 940.0, C).astype(np.float32)
    torch.manual_seed(0)
    batch = (torch.randn(BS, C, HW, HW), torch.zeros(BS, dtype=torch.long))

    without = extract_penultimate_features(model, iter([batch]), "cpu", max_samples=BS)[0]
    with_wl = extract_penultimate_features(model, iter([batch]), "cpu", max_samples=BS,
                                            wavelengths=wl)[0]
    assert not np.allclose(without, with_wl), \
        "wavelength forwarding had no effect - the positional-encoding branch was not taken"


# ----------------------------------------------------------------------------
# The sampling flaw that made the two surviving plots uninformative
# ----------------------------------------------------------------------------

def test_stratified_indices_cover_every_class_on_a_class_ordered_split():
    """`X_*.npy` is written capture by capture and is therefore class-ordered:
    the first 600 validation patches of the histology build are all one class.
    Taking the first N off an unshuffled loader - which is what the frozen
    extractor's `break` does - plots a single class and looks plausible."""
    from scripts.latent_space import _stratified_indices

    labels = np.concatenate([np.zeros(500), np.ones(50), np.full(2000, 2)]).astype(np.int64)
    path = "/tmp/_v18_labels_test.npy"
    np.save(path, labels)
    try:
        idx = _stratified_indices(path, max_samples=300, seed=0)
        chosen = labels[np.asarray(idx)]
        assert set(np.unique(chosen).tolist()) == {0, 1, 2}, "a class was missed entirely"
        assert len(idx) == 300
        assert len(set(idx)) == len(idx), "sampled with replacement"
        # The naive path, for contrast: the first 300 are one class.
        assert len(set(labels[:300].tolist())) == 1
    finally:
        import os
        os.remove(path)
