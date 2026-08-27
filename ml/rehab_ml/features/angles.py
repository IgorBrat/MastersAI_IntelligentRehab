"""Clinical joint angles for post-stroke upper-limb assessment"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterator

import numpy as np

from ..schema import Joint, PoseSequence, Side, joint_of
from .geometry import (
    BodyFrame,
    angle_between,
    gravity_up,
    joint_angle,
    pelvis_frame,
    project_onto_plane,
    signed_angle,
    unit,
)

# The plane of elevation is only defined in a BAND of shoulder elevation
PLANE_OF_ELEVATION_MIN_ELEVATION_DEG = 40.0
PLANE_OF_ELEVATION_MAX_ELEVATION_DEG = 140.0

#: Acromion rise per degree of humeral elevation, in torso lengths, above `SCAPULOHUMERAL_ONSET_DEG`
SCAPULOHUMERAL_RISE_PER_DEG = 0.00071

#: Elevation below which the scapula contributes little
SCAPULOHUMERAL_ONSET_DEG = 30.0


def _side_sign(side: Side) -> float:
    return 1.0 if side is Side.LEFT else -1.0


@dataclass
class AngleSeries:
    """Per-frame clinical angles for one working limb, plus trunk posture"""

    side: Side
    fps: float

    # -- working limb
    shoulder_elevation: np.ndarray
    shoulder_flexion: np.ndarray
    shoulder_abduction: np.ndarray
    plane_of_elevation: np.ndarray
    shoulder_rotation: np.ndarray
    elbow_flexion: np.ndarray

    # -- trunk posture (measured against gravity, not the body frame)
    trunk_forward_flexion: np.ndarray
    trunk_lateral_tilt: np.ndarray
    trunk_rotation: np.ndarray

    # -- scapular / girdle
    shoulder_height_working: np.ndarray
    shoulder_height_other: np.ndarray

    # -- end-effector, in body-frame torso-length units
    hand_forward: np.ndarray
    hand_up: np.ndarray
    hand_lateral: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.shoulder_elevation.shape[0])

    def trunk_lean_contralateral(self) -> np.ndarray:
        """Lateral lean *away from* the working arm, in degrees (+ = compensating)"""
        return -_side_sign(self.side) * self.trunk_lateral_tilt

    def shoulder_hike(
        self, baseline_frames: int = 5, *, scapular_correction: bool = True
    ) -> np.ndarray:
        """Scapular elevation *in excess of* the obligatory contribution to elevation"""
        n = max(1, min(baseline_frames, self.n_frames))
        baseline = float(np.median(self.shoulder_height_working[:n]))
        rise = self.shoulder_height_working - baseline
        if not scapular_correction:
            return rise
        obligatory = SCAPULOHUMERAL_RISE_PER_DEG * np.maximum(
            0.0, self.shoulder_elevation - SCAPULOHUMERAL_ONSET_DEG
        )
        return rise - obligatory

    def plane_deviation_from(self, target_plane_deg: float) -> np.ndarray:
        """Absolute departure from a prescribed plane of elevation, in degrees"""
        difference = (self.plane_of_elevation - target_plane_deg + 180.0) % 360.0 - 180.0
        gate = (
            self.shoulder_elevation >= PLANE_OF_ELEVATION_MIN_ELEVATION_DEG
        ) & (self.shoulder_elevation <= PLANE_OF_ELEVATION_MAX_ELEVATION_DEG)
        return np.where(gate, np.abs(difference), 0.0)

    def channel_names(self) -> tuple[str, ...]:
        """Names of the ``(T,)`` array fields, in declaration order"""
        return tuple(
            f.name for f in fields(self) if f.name not in ("side", "fps")
        )

    def __getitem__(self, name: str) -> np.ndarray:
        value = getattr(self, name)
        if not isinstance(value, np.ndarray):
            raise KeyError(f"{name!r} is not an angle channel")
        return value

    def items(self) -> Iterator[tuple[str, np.ndarray]]:
        for name in self.channel_names():
            yield name, getattr(self, name)

    def as_matrix(self) -> np.ndarray:
        """``(T, C)`` stacked channels, column order = :meth:`channel_names`"""
        return np.stack([getattr(self, n) for n in self.channel_names()], axis=-1)

    def slice(self, start: int, stop: int) -> "AngleSeries":
        kwargs = {"side": self.side, "fps": self.fps}
        for name in self.channel_names():
            kwargs[name] = getattr(self, name)[start:stop]
        return AngleSeries(**kwargs)


def compute_angles(seq: PoseSequence, side: Side) -> AngleSeries:
    """Computing the full clinical angle set for ``side`` of ``seq``"""
    j = seq.joints.astype(np.float64)
    frame = BodyFrame(j)
    up = gravity_up(j)
    sign = _side_sign(side)

    shoulder = j[:, joint_of(side, "SHOULDER"), :]
    elbow = j[:, joint_of(side, "ELBOW"), :]
    wrist = j[:, joint_of(side, "WRIST"), :]
    hand = j[:, joint_of(side, "HAND"), :]

    humerus = elbow - shoulder
    forearm = wrist - elbow
    down = -frame.longitudinal

    # shoulder
    elevation = angle_between(humerus, down)

    # Flexion: humerus in the sagittal plane
    hum_sag = project_onto_plane(humerus, frame.lateral)
    flexion = -signed_angle(down, hum_sag, frame.lateral)

    # Abduction: humerus in the frontal plane
    hum_fro = project_onto_plane(humerus, frame.anterior)
    abduction = sign * signed_angle(down, hum_fro, frame.anterior)

    # Plane of elevation: where the humerus sits in the transverse plane
    hum_tra = project_onto_plane(humerus, frame.longitudinal)
    plane = sign * signed_angle(frame.anterior, hum_tra, frame.longitudinal)

    # Axial rotation of the humerus
    hum_axis = unit(humerus)
    forearm_perp = project_onto_plane(forearm, hum_axis)
    reference = unit(project_onto_plane(frame.anterior, hum_axis))
    rotation = -sign * signed_angle(reference, forearm_perp, hum_axis)

    # Elbow
    elbow_flexion = 180.0 - joint_angle(shoulder, elbow, wrist)

    # Trunk, against gravity
    _, pelvis_lat, pelvis_ant = pelvis_frame(j)
    trunk_axis = frame.longitudinal

    trunk_sag = project_onto_plane(trunk_axis, pelvis_lat)
    up_sag = project_onto_plane(up, pelvis_lat)
    trunk_forward = signed_angle(up_sag, trunk_sag, pelvis_lat)

    trunk_fro = project_onto_plane(trunk_axis, pelvis_ant)
    up_fro = project_onto_plane(up, pelvis_ant)
    trunk_tilt = -signed_angle(up_fro, trunk_fro, pelvis_ant)

    # Axial rotation
    shoulder_line = j[:, Joint.L_SHOULDER, :] - j[:, Joint.R_SHOULDER, :]
    hip_line = j[:, Joint.L_HIP, :] - j[:, Joint.R_HIP, :]
    shoulder_tra = project_onto_plane(shoulder_line, up)
    hip_tra = project_onto_plane(hip_line, up)
    trunk_rotation = signed_angle(hip_tra, shoulder_tra, up)

    # Girdle heights and end-effector in TL
    scale = max(seq.torso_length(), 1e-3)
    local = frame.to_local(j) / scale
    shoulder_h_work = local[:, joint_of(side, "SHOULDER"), 1]
    shoulder_h_other = local[:, joint_of(side.other, "SHOULDER"), 1]

    hand_local = frame.to_local(hand) / scale

    return AngleSeries(
        side=side,
        fps=seq.fps,
        shoulder_elevation=elevation.astype(np.float32),
        shoulder_flexion=flexion.astype(np.float32),
        shoulder_abduction=abduction.astype(np.float32),
        plane_of_elevation=plane.astype(np.float32),
        shoulder_rotation=rotation.astype(np.float32),
        elbow_flexion=elbow_flexion.astype(np.float32),
        trunk_forward_flexion=trunk_forward.astype(np.float32),
        trunk_lateral_tilt=trunk_tilt.astype(np.float32),
        trunk_rotation=trunk_rotation.astype(np.float32),
        shoulder_height_working=shoulder_h_work.astype(np.float32),
        shoulder_height_other=shoulder_h_other.astype(np.float32),
        hand_forward=hand_local[:, 2].astype(np.float32),
        hand_up=hand_local[:, 1].astype(np.float32),
        hand_lateral=(_side_sign(side) * hand_local[:, 0]).astype(np.float32),
    )


ANGLE_CHANNELS: tuple[str, ...] = (
    "shoulder_elevation",
    "shoulder_flexion",
    "shoulder_abduction",
    "plane_of_elevation",
    "shoulder_rotation",
    "elbow_flexion",
    "trunk_forward_flexion",
    "trunk_lateral_tilt",
    "trunk_rotation",
    "shoulder_height_working",
    "shoulder_height_other",
    "hand_forward",
    "hand_up",
    "hand_lateral",
)
N_ANGLE_CHANNELS = len(ANGLE_CHANNELS)
