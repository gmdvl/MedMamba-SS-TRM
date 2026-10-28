# -*- coding: utf-8 -*-
"""
hsi_augment/qc/memorization.py
==============================
R6, memorization (Plan §Phase 8b, issue #28): exact byte-hash matches against
every train patch, and the distance to the nearest training patch.

Why two checks
--------------
A generator that reproduces a training patch byte for byte is caught by the
hash (`HashIndex`: blake2b-128 of the float16 bytes, all 2.45M train rows in
~100 MB, sorted, binary-searched). One that reproduces it up to a little noise
is not, so the nearest-neighbour distance must also stay above what real data
shows: at least the q1 of the real CROSS-CAPTURE nearest-neighbour distances
(a real patch's nearest patch in another capture; patches of the same capture
are spatial neighbours and closer than any copy needs to be). Both are needed
and both are recorded per sample.

How the nearest neighbour is found
----------------------------------
`knn_search` finds the top `candidates` (10) in the PCA-64 space with
`torch.cdist`, on the GPU when there is one and on the CPU otherwise. Query and
bank are both chunked so a block never exceeds `block_elems` distances
(2^24 = 64 MB of float32), and a running top-k is kept across bank blocks, so
RAM and VRAM stay bounded whatever the sizes. `torch.cdist` computes
||a||^2 + ||b||^2 - 2ab for speed, which is inaccurate exactly where a
memorization check looks (near-zero distances), and PCA drops detail, so the
candidates are RE-RANKED with the exact distance in the full z-scored space
(`exact_rerank`): the root mean square difference of the flattened, band
z-scored patches, in float64.

Alaa-style authenticity (Alaa et al. 2022) is reported, not gated: a sample is
authentic when it is farther from its nearest real patch than that real
patch is from ITS nearest real neighbour (here: cross-capture).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, NamedTuple, Optional, Tuple

import numpy as np

DEFAULT_BLOCK_ELEMS = 1 << 24
DEFAULT_CANDIDATES = 10


def resolve_device(device: Optional[str] = "auto") -> str:
    """'auto' -> 'cuda' when torch sees a GPU, else 'cpu' (the CPU fallback)."""
    if device in (None, "auto"):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


# ============================================================================
# exact byte hashes
# ============================================================================

@dataclass
class HashIndex:
    """Sorted blake2b-128 hashes of train patches, with each row's provenance."""
    h1: np.ndarray        # uint64, ascending (then h2)
    h2: np.ndarray        # uint64
    row: np.ndarray       # int64 train row
    patient: np.ndarray   # int64
    capture: np.ndarray   # int64

    @classmethod
    def build(cls, hashes: np.ndarray, row: np.ndarray, patient: np.ndarray, capture: np.ndarray) -> "HashIndex":
        hashes = np.asarray(hashes, dtype=np.uint64).reshape(-1, 2)
        order = np.lexsort((hashes[:, 1], hashes[:, 0]))
        return cls(h1=hashes[order, 0].copy(), h2=hashes[order, 1].copy(),
                   row=np.asarray(row, dtype=np.int64)[order], patient=np.asarray(patient, dtype=np.int64)[order],
                   capture=np.asarray(capture, dtype=np.int64)[order])

    def __len__(self) -> int:
        return int(self.h1.shape[0])

    def lookup(self, hashes: np.ndarray, *, exclude_patients: Optional[Iterable[int]] = None,
               exclude_capture: Optional[np.ndarray] = None) -> np.ndarray:
        """The train row each query hash matches exactly, or -1. Matches with a
        row of `exclude_patients` (a fold's held-out patients) or of the
        query's own capture (`exclude_capture[i]`) do not count."""
        q = np.asarray(hashes, dtype=np.uint64).reshape(-1, 2)
        out = np.full(len(q), -1, dtype=np.int64)
        if len(self) == 0 or len(q) == 0:
            return out
        left = np.searchsorted(self.h1, q[:, 0], side="left")
        right = np.searchsorted(self.h1, q[:, 0], side="right")
        excl = None if exclude_patients is None else np.asarray(sorted(int(p) for p in exclude_patients),
                                                                 dtype=np.int64)
        for i in np.flatnonzero(right > left):      # rare: real hits (or 64-bit prefix collisions)
            for j in range(left[i], right[i]):
                if self.h2[j] != q[i, 1]:
                    continue
                if excl is not None and len(excl) and np.isin(self.patient[j], excl):
                    continue
                if exclude_capture is not None and self.capture[j] == exclude_capture[i]:
                    continue
                out[i] = self.row[j]
                break
        return out

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {"h1": self.h1, "h2": self.h2, "row": self.row, "patient": self.patient, "capture": self.capture}

    @classmethod
    def from_arrays(cls, d: Mapping[str, np.ndarray]) -> "HashIndex":
        return cls(**{k: np.asarray(d[k]) for k in ("h1", "h2", "row", "patient", "capture")})


# ============================================================================
# chunked k-nearest-neighbour search (approximate distances)
# ============================================================================

def knn_search(query: np.ndarray, bank: np.ndarray, k: int, *, device: Optional[str] = "auto",
               block_elems: int = DEFAULT_BLOCK_ELEMS, bank_valid: Optional[np.ndarray] = None,
               query_groups: Optional[np.ndarray] = None,
               bank_groups: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """The `k` nearest bank rows of every query row, by `torch.cdist`.

    `bank_valid` (bool per bank row) removes rows (a fold's held-out patients);
    with `query_groups` and `bank_groups`, bank rows of the query's own group
    (its capture) are skipped. Returns (dist float32 [n, k], idx int64 [n, k]),
    ascending; a slot with no eligible row has dist inf and idx -1. Distances
    are cdist's (matrix-product form): use `exact_rerank` where they matter."""
    import torch
    q_np = np.ascontiguousarray(query, dtype=np.float32)
    b_np = np.ascontiguousarray(bank, dtype=np.float32)
    nq, nb = len(q_np), len(b_np)
    out_d = np.full((nq, k), np.inf, dtype=np.float32)
    out_i = np.full((nq, k), -1, dtype=np.int64)
    if nq == 0 or nb == 0 or k <= 0:
        return out_d, out_i
    dev = torch.device(resolve_device(device))
    b = torch.from_numpy(b_np).to(dev)
    valid = None if bank_valid is None else torch.from_numpy(np.asarray(bank_valid, dtype=bool)).to(dev)
    bg = None if bank_groups is None else torch.from_numpy(np.asarray(bank_groups, dtype=np.int64)).to(dev)
    qg_all = None if query_groups is None else np.asarray(query_groups, dtype=np.int64)
    q_rows = int(max(1, min(nq, block_elems // max(min(nb, block_elems), 1))))
    b_cols = int(max(1, min(nb, block_elems // q_rows)))
    kk = min(k, nb)
    with torch.no_grad():
        for qa in range(0, nq, q_rows):
            qb = min(qa + q_rows, nq)
            qt = torch.from_numpy(q_np[qa:qb]).to(dev)
            qg = None if qg_all is None else torch.from_numpy(qg_all[qa:qb]).to(dev)
            best_d = torch.full((qb - qa, 0), float("inf"), device=dev)
            best_i = torch.full((qb - qa, 0), -1, dtype=torch.int64, device=dev)
            for ba in range(0, nb, b_cols):
                bb = min(ba + b_cols, nb)
                d = torch.cdist(qt, b[ba:bb])
                if valid is not None:
                    d = d.masked_fill(~valid[ba:bb][None, :], float("inf"))
                if qg is not None and bg is not None:
                    d = d.masked_fill(qg[:, None] == bg[None, ba:bb], float("inf"))
                idx = torch.arange(ba, bb, device=dev).expand(qb - qa, -1)
                cand_d = torch.cat([best_d, d], dim=1)
                cand_i = torch.cat([best_i, idx], dim=1)
                top = torch.topk(cand_d, min(kk, cand_d.shape[1]), dim=1, largest=False, sorted=True)
                best_d, best_i = top.values, torch.gather(cand_i, 1, top.indices)
            bd, bi = best_d.cpu().numpy(), best_i.cpu().numpy()
            bi = np.where(np.isfinite(bd), bi, -1)
            out_d[qa:qb, :bd.shape[1]] = bd
            out_i[qa:qb, :bi.shape[1]] = bi
    return out_d, out_i


# ============================================================================
# exact re-rank
# ============================================================================

def exact_rerank(query_z: np.ndarray, cand_idx: np.ndarray,
                 bank_z: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Exact RMS distance in the full z space from each query to each of its
    candidates; returns (best distance float64 [n], best bank index [n], -1 if
    none). `bank_z(sorted unique bank indices)` returns their z-flat rows.
    Ties keep the earlier (PCA-closer) candidate."""
    qz = np.asarray(query_z, dtype=np.float32)
    cand = np.asarray(cand_idx, dtype=np.int64)
    n, k = cand.shape
    dist = np.full((n, k), np.inf)
    ok = cand >= 0
    if ok.any():
        uniq = np.unique(cand[ok])
        zb = np.asarray(bank_z(uniq), dtype=np.float32)
        pos = np.searchsorted(uniq, np.where(ok, cand, uniq[0]))
        dim = qz.shape[1]
        for j in range(k):
            v = ok[:, j]
            if v.any():
                diff = qz[v] - zb[pos[v, j]]
                dist[v, j] = np.sqrt(np.einsum("ij,ij->i", diff, diff, dtype=np.float64) / dim)
    best = np.argmin(dist, axis=1)
    rows = np.arange(n)
    best_d = dist[rows, best]
    best_i = np.where(np.isfinite(best_d), cand[rows, best], -1)
    return best_d, best_i


class NNResult(NamedTuple):
    dist: np.ndarray          # float64 [n], exact RMS z distance (inf: no eligible bank row)
    idx: np.ndarray           # int64 [n], bank index (-1: none)
    approx_dist: np.ndarray   # float32 [n], the PCA-space distance of the chosen candidate


def nearest_neighbours(query_pca: np.ndarray, query_z: Callable[[np.ndarray], np.ndarray],
                       bank_pca: np.ndarray, bank_z: Callable[[np.ndarray], np.ndarray], *,
                       candidates: int = DEFAULT_CANDIDATES, query_chunk: int = 512,
                       device: Optional[str] = "auto", block_elems: int = DEFAULT_BLOCK_ELEMS,
                       bank_valid: Optional[np.ndarray] = None, query_groups: Optional[np.ndarray] = None,
                       bank_groups: Optional[np.ndarray] = None) -> NNResult:
    """PCA top-`candidates` search, then the exact re-rank, `query_chunk`
    queries at a time. `query_z(query indices)` and `bank_z(bank indices)`
    return z-flat rows on demand, so neither side is ever held whole."""
    qp = np.asarray(query_pca, dtype=np.float32)
    n = len(qp)
    dist = np.full(n, np.inf)
    idx = np.full(n, -1, dtype=np.int64)
    approx = np.full(n, np.inf, dtype=np.float32)
    for a in range(0, n, max(1, int(query_chunk))):
        b = min(a + int(query_chunk), n)
        qg = None if query_groups is None else np.asarray(query_groups)[a:b]
        cd, ci = knn_search(qp[a:b], bank_pca, candidates, device=device, block_elems=block_elems,
                            bank_valid=bank_valid, query_groups=qg, bank_groups=bank_groups)
        d, i = exact_rerank(query_z(np.arange(a, b)), ci, bank_z)
        dist[a:b], idx[a:b] = d, i
        hit = i >= 0
        pos = np.argmax(ci == i[:, None], axis=1)
        approx[a:b] = np.where(hit, cd[np.arange(b - a), pos], np.inf)
    return NNResult(dist, idx, approx)
