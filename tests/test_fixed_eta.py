# -*- coding: utf-8 -*-
"""tests/test_fixed_eta.py - the fixed-eta control must give every retained
band exactly the positional code it had at training time, while the default
(eta = C, self-normalized range) must not."""
import numpy as np
import torch

from medmamba_ss_trm import SpectralTokenizer
from scripts.eval_band_decimation_fixed_eta import fix_encoding, training_range
from scripts.eval_band_decimation import decimate_indices

WL = np.array([400.5, 430.0, 463.2, 500.0, 535.1, 562.0, 577.3, 633.3,
               852.3, 870.0, 900.0, 938.2], dtype=np.float32)


def _pe(tok, wl, sensor_range=None):
    v = torch.zeros(1, len(wl))
    return tok._positional_encoding(v, torch.tensor(wl), sensor_range, torch.float32)[0]


def test_fixed_eta_reproduces_training_codes():
    tok = SpectralTokenizer(d_token=16, fusion="concat_mlp", wavelength_encoding_scale=None)
    full = _pe(tok, WL)                                   # training: eta = 12, own range
    idx = decimate_indices(len(WL), 4)
    drifted = _pe(tok, WL[idx])                           # default zero-shot: eta = 4
    assert not torch.allclose(drifted, full[idx], atol=1e-4)

    assert fix_encoding(tok, len(WL)) == 1
    fixed = _pe(tok, WL[idx], sensor_range=training_range(WL))
    assert torch.allclose(fixed, full[idx], atol=1e-5)


def test_explicit_scale_is_left_alone():
    tok = SpectralTokenizer(d_token=16, fusion="concat_mlp", wavelength_encoding_scale=1000.0)
    assert fix_encoding(tok, 32) == 0 and tok.wavelength_encoding_scale == 1000.0
