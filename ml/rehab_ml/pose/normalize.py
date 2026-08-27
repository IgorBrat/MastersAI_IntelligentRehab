from __future__ import annotations

import numpy as np

from ..schema import Joint, PoseSequence, SKELETON_PARENTS
from ..features.kinematics import smooth_signal


def fill_gaps(
    seq: PoseSequence,
    *,
    confidence_threshold: float = 0.3,
    max_gap_s: float = 0.5,
) -> PoseSequence:
    """Linearly interpolate low-confidence or non-finite joint positions"""
    joints = seq.joints.astype(np.float64).copy()
    conf = seq.confidence.copy()
    n_frames = seq.n_frames
    max_gap = max(1, int(round(max_gap_s * seq.fps)))
    t = np.arange(n_frames)

    for j in range(joints.shape[1]):
        bad = (conf[:, j] < confidence_threshold) | ~np.isfinite(joints[:, j, :]).all(axis=-1)
        if not bad.any():
            continue
        good = ~bad
        if good.sum() < 2:
            # Nothing to interpolate from: hold the joint at the pelvis so downstream maths stays finite, and leave confidence at zero
            joints[:, j, :] = joints[:, Joint.PELVIS, :]
            conf[:, j] = 0.0
            continue
        for axis in range(3):
            joints[bad, j, axis] = np.interp(t[bad], t[good], joints[good, j, axis])
        # Mark frames belonging to over-long gaps as still untrusted
        for start, stop in _runs(bad):
            if stop - start > max_gap:
                conf[start:stop, j] = 0.0

    out = PoseSequence(joints=joints, fps=seq.fps, confidence=conf, meta=dict(seq.meta))
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` index ranges of contiguous True values"""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def denoise(seq: PoseSequence, *, cutoff_hz: float = 6.0) -> PoseSequence:
    """Low-pass every joint coordinate independently"""
    if seq.n_frames < 5:
        return seq
    flat = seq.joints.reshape(seq.n_frames, -1)
    smoothed = smooth_signal(flat, seq.fps, cutoff_hz=cutoff_hz, axis=0)
    return seq.with_joints(smoothed.reshape(seq.joints.shape).astype(np.float32))


def center_and_scale(seq: PoseSequence, *, fixed_scale: float | None = None) -> PoseSequence:
    """Translate the pelvis to the origin and rescale to torso-length units"""
    joints = seq.joints.astype(np.float64)
    pelvis = joints[:, Joint.PELVIS : Joint.PELVIS + 1, :]
    centred = joints - pelvis
    scale = fixed_scale if fixed_scale is not None else seq.torso_length()
    scale = max(float(scale), 1e-3)
    out = seq.with_joints((centred / scale).astype(np.float32))
    out.meta["scale_torso_lengths"] = scale
    return out


def enforce_bone_lengths(seq: PoseSequence, *, strength: float = 1.0) -> PoseSequence:
    """Project joints so each bone keeps its median length over the sequence"""
    joints = seq.joints.astype(np.float64).copy()
    order = _topological_joints()
    for child in order:
        parent = SKELETON_PARENTS[child]
        if parent == child:
            continue
        vec = joints[:, child, :] - joints[:, parent, :]
        length = np.linalg.norm(vec, axis=-1, keepdims=True)
        target = float(np.median(length))
        if target <= 1e-6:
            continue
        corrected = joints[:, parent, :] + vec / np.maximum(length, 1e-8) * target
        joints[:, child, :] = (1.0 - strength) * joints[:, child, :] + strength * corrected
    return seq.with_joints(joints.astype(np.float32))


def _topological_joints() -> list[Joint]:
    """Joints ordered so every parent precedes its children"""
    ordered: list[Joint] = []
    remaining = set(SKELETON_PARENTS)
    while remaining:
        progressed = False
        for joint in sorted(remaining, key=int):
            parent = SKELETON_PARENTS[joint]
            if parent == joint or parent in ordered:
                ordered.append(joint)
                remaining.discard(joint)
                progressed = True
        if not progressed:
            raise RuntimeError("skeleton parent table contains a cycle")
    return ordered


def preprocess(
    seq: PoseSequence,
    *,
    confidence_threshold: float = 0.3,
    cutoff_hz: float = 6.0,
    rigid_bones: bool = True,
    scale_normalise: bool = True,
) -> PoseSequence:
    """The standard pipeline: fill gaps -> denoise -> rigidify -> centre/scale"""
    out = fill_gaps(seq, confidence_threshold=confidence_threshold)
    out = denoise(out, cutoff_hz=cutoff_hz)
    if rigid_bones:
        out = enforce_bone_lengths(out)
    if scale_normalise:
        out = center_and_scale(out)
    out.meta["preprocessed"] = True
    return out
