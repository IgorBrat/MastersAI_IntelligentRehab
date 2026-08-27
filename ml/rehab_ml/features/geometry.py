from __future__ import annotations

import numpy as np

from ..schema import Joint

EPS = 1e-8


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalising to unit length"""
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(norm, EPS)


def angle_between(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unsigned angle in degrees between two (stacks of) vectors"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.sum(a * b, axis=axis)
    if a.shape[axis] != 3:
        raise ValueError("angle_between requires 3-vectors")
    cross_norm = np.linalg.norm(np.cross(a, b, axis=axis), axis=axis)
    return np.degrees(np.arctan2(cross_norm, dot))


def signed_angle(a: np.ndarray, b: np.ndarray, normal: np.ndarray, axis: int = -1) -> np.ndarray:
    """Angle from ``a`` to ``b`` in degrees, signed by right-hand rule about ``normal``"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = unit(np.asarray(normal, dtype=np.float64), axis=axis)
    cross = np.cross(a, b, axis=axis)
    sin_term = np.sum(cross * n, axis=axis)
    cos_term = np.sum(a * b, axis=axis)
    return np.degrees(np.arctan2(sin_term, cos_term))


def joint_angle(proximal: np.ndarray, vertex: np.ndarray, distal: np.ndarray) -> np.ndarray:
    """Interior angle at ``vertex`` in degrees, e.g. elbow from shoulder/elbow/wrist"""
    return angle_between(proximal - vertex, distal - vertex)


def project_onto_plane(v: np.ndarray, normal: np.ndarray, axis: int = -1) -> np.ndarray:
    """Removing the component of ``v`` along ``normal``"""
    n = unit(np.asarray(normal, dtype=np.float64), axis=axis)
    return v - np.sum(v * n, axis=axis, keepdims=True) * n


class BodyFrame:
    """Per-frame orthonormal anatomical basis attached to the pelvis and trunk"""

    __slots__ = ("lateral", "longitudinal", "anterior", "origin")

    def __init__(self, joints: np.ndarray) -> None:
        l_hip = joints[:, Joint.L_HIP, :].astype(np.float64)
        r_hip = joints[:, Joint.R_HIP, :].astype(np.float64)
        pelvis = joints[:, Joint.PELVIS, :].astype(np.float64)
        neck = joints[:, Joint.NECK, :].astype(np.float64)

        lateral_raw = l_hip - r_hip
        longitudinal = unit(neck - pelvis)
        lateral = unit(project_onto_plane(lateral_raw, longitudinal))
        anterior = unit(np.cross(lateral, longitudinal))

        self.lateral = lateral
        self.longitudinal = longitudinal
        self.anterior = anterior
        self.origin = pelvis

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """World points in frame coordinates"""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 3:
            rel = pts - self.origin[:, None, :]
            basis = np.stack([self.lateral, self.longitudinal, self.anterior], axis=1)
            return np.einsum("tij,tkj->tki", basis, rel)
        rel = pts - self.origin
        return np.stack(
            [
                np.sum(rel * self.lateral, axis=-1),
                np.sum(rel * self.longitudinal, axis=-1),
                np.sum(rel * self.anterior, axis=-1),
            ],
            axis=-1,
        )

def pelvis_frame(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gravity-referenced frame attached to the pelvis: ``(up, lateral, anterior)``"""
    up = gravity_up(joints)
    hip_axis = joints[:, Joint.L_HIP, :].astype(np.float64) - joints[
        :, Joint.R_HIP, :
    ].astype(np.float64)
    lateral = unit(project_onto_plane(hip_axis, up))
    anterior = unit(np.cross(lateral, up))
    return up, lateral, anterior


def gravity_up(joints: np.ndarray) -> np.ndarray:
    """Returning the world 'up' direction as a ``(T, 3)`` broadcast array"""
    t = joints.shape[0]
    up = np.zeros((t, 3), dtype=np.float64)
    up[:, 1] = 1.0
    return up
