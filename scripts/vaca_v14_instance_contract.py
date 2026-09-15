#!/usr/bin/env python3
"""VACA V1.4 multi-cow instance/temporal fusion contract.

This module deliberately does NOT run a detector or VOS model. It defines the
model-independent rules used after RF-DETR/RTMDet instance masks and Cutie
propagation so one cow cannot silently become another cow during contact.

Production principles:
- one visible pixel has at most one owner;
- spatial detector and temporal propagator are independent evidence sources;
- unresolved overlap or identity disagreement => REVIEW, never invented Cow ID;
- passage track_id is transient; permanent cow_id is assigned by a later Re-ID gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math
import numpy as np


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"
    VISIBLE = "VISIBLE"
    OCCLUDED = "OCCLUDED"
    REACQUIRE = "REACQUIRE"
    LOST = "LOST"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class InstanceEvidence:
    instance_id: int
    mask: np.ndarray
    score: float
    centroid_xy: tuple[float, float]
    area: float
    orientation_deg: Optional[float] = None
    appearance: Optional[np.ndarray] = None


@dataclass(frozen=True)
class FusionDecision:
    owner_map: np.ndarray
    conflict_mask: np.ndarray
    review_instance_ids: tuple[int, ...]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))


def association_cost(prev: InstanceEvidence, cur: InstanceEvidence,
                     diag_px: float, predicted_centroid: Optional[tuple[float, float]] = None) -> float:
    """Lower is better. Returns inf when an association violates hard gates."""
    iou = mask_iou(prev.mask, cur.mask)
    px, py = predicted_centroid or prev.centroid_xy
    cx, cy = cur.centroid_xy
    motion = math.hypot(cx-px, cy-py) / max(diag_px, 1.0)
    area_ratio = max(prev.area, cur.area) / max(min(prev.area, cur.area), 1.0)
    app = cosine_similarity(prev.appearance, cur.appearance)

    if motion > 0.30:
        return math.inf
    if area_ratio > 3.0 and iou < 0.10:
        return math.inf
    if iou < 0.02 and (app is None or app < 0.70):
        return math.inf

    area_penalty = min(abs(math.log(max(area_ratio, 1e-6))) / math.log(3.0), 1.0)
    app_penalty = 0.5 if app is None else (1.0 - max(-1.0, min(1.0, app))) / 2.0
    return 0.50*(1.0-iou) + 0.20*min(motion/0.30, 1.0) + 0.10*area_penalty + 0.20*app_penalty


def enforce_visible_pixel_ownership(instances: list[InstanceEvidence],
                                    min_score: float = 0.35,
                                    conflict_margin: float = 0.08) -> FusionDecision:
    """Resolve overlapping visible masks without silently double-owning pixels.

    Pixels with competing scores closer than conflict_margin are set to 0 (unowned)
    and all involved instances are flagged REVIEW. This is deliberately conservative.
    """
    valid = [x for x in instances if x.score >= min_score and np.any(x.mask)]
    if not valid:
        shape = instances[0].mask.shape if instances else (0, 0)
        return FusionDecision(np.zeros(shape, np.int32), np.zeros(shape, bool), ())

    shape = valid[0].mask.shape
    scores = np.full((len(valid), *shape), -np.inf, dtype=np.float32)
    for k, inst in enumerate(valid):
        if inst.mask.shape != shape:
            raise ValueError("all masks must have same shape")
        scores[k][inst.mask.astype(bool)] = float(inst.score)

    order = np.argsort(scores, axis=0)
    best_k = order[-1]
    best = np.take_along_axis(scores, best_k[None, ...], axis=0)[0]
    second_k = order[-2] if len(valid) > 1 else np.zeros(shape, dtype=int)
    second = (np.take_along_axis(scores, second_k[None, ...], axis=0)[0]
              if len(valid) > 1 else np.full(shape, -np.inf, dtype=np.float32))
    occupied = np.isfinite(best)
    finite_pair = occupied & np.isfinite(second)
    gap = np.full(shape, np.inf, dtype=np.float32)
    gap[finite_pair] = best[finite_pair] - second[finite_pair]
    conflict = finite_pair & (gap < conflict_margin)

    owner = np.zeros(shape, np.int32)
    id_lut = np.asarray([x.instance_id for x in valid], dtype=np.int32)
    owner[occupied & ~conflict] = id_lut[best_k[occupied & ~conflict]]

    review = set()
    if np.any(conflict):
        ys, xs = np.nonzero(conflict)
        for y, x in zip(ys, xs):
            review.add(valid[int(best_k[y, x])].instance_id)
            review.add(valid[int(second_k[y, x])].instance_id)
    return FusionDecision(owner, conflict, tuple(sorted(review)))


def bidirectional_consensus(forward_mask: np.ndarray, backward_mask: np.ndarray,
                            min_iou: float = 0.70) -> tuple[np.ndarray, bool, float]:
    """Conservative Cutie forward/backward consensus."""
    score = mask_iou(forward_mask, backward_mask)
    if score < min_iou:
        return np.logical_and(forward_mask, backward_mask), True, score
    return np.logical_or(forward_mask, backward_mask), False, score


def _self_test() -> None:
    a = np.zeros((8, 8), bool)
    a[1:5, 1:5] = 1
    b = np.zeros((8, 8), bool)
    b[3:7, 3:7] = 1
    e1 = InstanceEvidence(1, a, 0.90, (2.5, 2.5), 16)
    e2 = InstanceEvidence(2, b, 0.88, (4.5, 4.5), 16)
    out = enforce_visible_pixel_ownership([e1, e2], conflict_margin=0.08)
    assert out.conflict_mask.sum() == 4
    assert out.review_instance_ids == (1, 2)
    mask, review, iou = bidirectional_consensus(a, a.copy())
    assert not review and iou == 1.0 and mask.sum() == 16
    print("SELF TEST PASS: ownership + bidirectional consensus")


if __name__ == "__main__":
    _self_test()
