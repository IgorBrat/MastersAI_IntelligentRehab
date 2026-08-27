from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..exercises.base import ExerciseDefinition
from ..exercises.stroke_upper_limb import PROTOCOL
from ..features.angles import SCAPULOHUMERAL_ONSET_DEG, SCAPULOHUMERAL_RISE_PER_DEG
from ..schema import Joint, N_JOINTS, PoseSequence, Side
from ..taxonomy import ErrorType


TORSO = 0.50
HIP_HALF_WIDTH = 0.09
SHOULDER_HALF_WIDTH = 0.18
SPINE_MID_HEIGHT = 0.25
NECK_HEIGHT = TORSO
HEAD_HEIGHT = 0.72
HUMERUS = 0.30
FOREARM = 0.26
HAND = 0.08
THIGH = 0.42
SHANK = 0.40


def _rotation(axis: np.ndarray, angle_deg: float | np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrix about a unit ``axis``"""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    theta = np.radians(np.atleast_1d(np.asarray(angle_deg, dtype=np.float64)))
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    eye = np.eye(3)
    out = (
        eye[None, :, :]
        + np.sin(theta)[:, None, None] * k[None, :, :]
        + (1.0 - np.cos(theta))[:, None, None] * (k @ k)[None, :, :]
    )
    return out


def _side_sign(side: Side) -> float:
    return 1.0 if side is Side.LEFT else -1.0


def minimum_jerk(n: int) -> np.ndarray:
    """Normalised minimum-jerk position profile on ``[0, 1]``"""
    t = np.linspace(0.0, 1.0, max(int(n), 2))
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def fragmented_profile(n: int, n_submovements: int, *, overlap: float = 0.25) -> np.ndarray:
    """A movement decomposed into overlapping submovements"""
    n = max(int(n), 2)
    k = max(int(n_submovements), 1)
    if k == 1:
        return minimum_jerk(n)

    t = np.linspace(0.0, 1.0, n)
    total = np.zeros(n, dtype=np.float64)
    span = 1.0 / (1.0 + (k - 1) * (1.0 - overlap))
    amplitudes = np.linspace(1.0, 0.45, k)
    amplitudes = amplitudes / amplitudes.sum()
    for i in range(k):
        start = i * span * (1.0 - overlap)
        end = min(start + span, 1.0)
        local = np.clip((t - start) / max(end - start, 1e-9), 0.0, 1.0)
        total += amplitudes[i] * (10 * local**3 - 15 * local**4 + 6 * local**5)
    return total / max(total[-1], 1e-9)


def repetition_profile(
    fps: float,
    concentric_s: float,
    hold_s: float,
    eccentric_s: float,
    *,
    n_submovements: int = 1,
    return_fraction: float = 1.0,
    n_reps: int = 1,
    rest_s: float = 0.0,
) -> np.ndarray:
    """``n_reps`` up-hold-down repetitions, normalised so the peak is ``1.0``"""
    n_up = max(int(round(concentric_s * fps)), 2)
    n_hold = max(int(round(hold_s * fps)), 0)
    n_down = max(int(round(eccentric_s * fps)), 2)
    n_rest = max(int(round(rest_s * fps)), 0)

    floor = 1.0 - float(np.clip(return_fraction, 0.0, 1.0))
    rise_shape = fragmented_profile(n_up, n_submovements)
    fall_shape = minimum_jerk(n_down)

    segments: list[np.ndarray] = []
    start = 0.0
    for _ in range(max(int(n_reps), 1)):
        segments.append(start + (1.0 - start) * rise_shape)
        if n_hold:
            segments.append(np.ones(n_hold, dtype=np.float64))
        segments.append(1.0 - (1.0 - floor) * fall_shape)
        if n_rest:
            segments.append(np.full(n_rest, floor, dtype=np.float64))
        start = floor
    return np.concatenate(segments)


@dataclass
class MovementParams:
    """One synthetic repetition"""

    exercise: ExerciseDefinition
    side: Side = Side.RIGHT

    # Peak of the exercise's driving angle, in degrees
    peak_deg: float = 90.0

    concentric_s: float = 3.0
    hold_s: float = 2.0
    eccentric_s: float = 3.0
    fps: float = 30.0

    # Error injection

    # Contralateral lateral trunk lean at peak -> TRUNK_LATERAL_LEAN
    trunk_lean_deg: float = 0.0

    # Forward trunk flexion at peak -> TRUNK_FORWARD_FLEXION
    trunk_flexion_deg: float = 0.0

    # Axial trunk rotation at peak -> TRUNK_ROTATION
    trunk_rotation_deg: float = 0.0

    # Acromion rise at peak, in torso lengths -> SHOULDER_HIKING
    shoulder_hike_tl: float = 0.0

    # Elbow flexion coupled to elevation -> ELBOW_FLEXION_SYNERGY
    elbow_synergy_deg: float = 0.0

    # Departure from the prescribed plane at peak -> OUT_OF_PLANE_DRIFT
    plane_drift_deg: float = 0.0

    # Above 1 -> JERKY_MOVEMENT
    n_submovements: int = 1

    # Below 1 -> INCOMPLETE_RETURN
    return_fraction: float = 1.0

    # Whole-body yaw about the vertical, in degrees. 0 faces the camera, 90 is side-on
    subject_yaw_deg: float = 0.0

    # Gaussian noise added to every joint, in metres
    noise_m: float = 0.0
    seed: int | None = 0

    @property
    def duration_s(self) -> float:
        return self.concentric_s + self.hold_s + self.eccentric_s

    def injected_errors(self) -> set[ErrorType]:
        """Which errors this parameter set is *intended* to produce"""
        out: set[ErrorType] = set()
        if self.trunk_lean_deg > 0:
            out.add(ErrorType.TRUNK_LATERAL_LEAN)
        if self.trunk_flexion_deg > 0:
            out.add(ErrorType.TRUNK_FORWARD_FLEXION)
        if self.trunk_rotation_deg != 0:
            out.add(ErrorType.TRUNK_ROTATION)
        if self.shoulder_hike_tl > 0:
            out.add(ErrorType.SHOULDER_HIKING)
        if self.elbow_synergy_deg > 0:
            out.add(ErrorType.ELBOW_FLEXION_SYNERGY)
        if self.plane_drift_deg != 0:
            out.add(ErrorType.OUT_OF_PLANE_DRIFT)
        if self.n_submovements > 1:
            out.add(ErrorType.JERKY_MOVEMENT)
        if self.return_fraction < 1.0:
            out.add(ErrorType.INCOMPLETE_RETURN)
        return out


@dataclass
class _Drive:
    """Per-frame joint-angle drive signals produced by an exercise's pattern"""

    elevation: np.ndarray
    plane: np.ndarray
    elbow: np.ndarray
    rotation: np.ndarray


def _drive_signals(params: MovementParams, profile: np.ndarray) -> _Drive:
    """Mapping the normalised profile onto the four arm degrees of freedom"""
    ex = params.exercise
    zeros = np.zeros_like(profile)
    peak = params.peak_deg

    if ex.primary_channel == "shoulder_abduction":
        return _Drive(profile * peak, zeros + 90.0, zeros, zeros)
    if ex.primary_channel == "shoulder_flexion":
        return _Drive(profile * peak, zeros, zeros, zeros)
    if ex.primary_channel == "elbow_flexion":
        return _Drive(zeros, zeros, profile * peak, zeros)
    if ex.primary_channel == "shoulder_rotation":
        return _Drive(zeros, zeros, zeros + 90.0, profile * peak)
    if ex.primary_channel == "hand_forward":
        return _Drive(profile * peak, zeros, 70.0 * (1.0 - profile), zeros)
    if ex.primary_channel == "hand_up":
        return _Drive(profile * peak, profile * 25.0, profile * 130.0, zeros)
    raise ValueError(f"no synthetic drive defined for channel {ex.primary_channel!r}")


def build_skeleton(
    elevation: np.ndarray,
    plane: np.ndarray,
    elbow: np.ndarray,
    rotation: np.ndarray,
    side: Side,
    *,
    trunk_lean: np.ndarray | None = None,
    trunk_flexion: np.ndarray | None = None,
    trunk_rotation: np.ndarray | None = None,
    shoulder_hike: np.ndarray | None = None,
    subject_yaw_deg: float = 0.0,
) -> np.ndarray:
    """Forward kinematics: joint angles -> ``(T, 18, 3)`` canonical positions"""
    n = len(elevation)
    sign = _side_sign(side)
    zeros = np.zeros(n, dtype=np.float64)

    lean = zeros if trunk_lean is None else np.asarray(trunk_lean, dtype=np.float64)
    flex = zeros if trunk_flexion is None else np.asarray(trunk_flexion, dtype=np.float64)
    twist = zeros if trunk_rotation is None else np.asarray(trunk_rotation, dtype=np.float64)
    hike = zeros if shoulder_hike is None else np.asarray(shoulder_hike, dtype=np.float64)

    joints = np.zeros((n, N_JOINTS, 3), dtype=np.float64)

    joints[:, Joint.PELVIS] = 0.0
    joints[:, Joint.L_HIP] = [HIP_HALF_WIDTH, 0.0, 0.0]
    joints[:, Joint.R_HIP] = [-HIP_HALF_WIDTH, 0.0, 0.0]
    for hip, knee, ankle, x in (
        (Joint.L_HIP, Joint.L_KNEE, Joint.L_ANKLE, HIP_HALF_WIDTH),
        (Joint.R_HIP, Joint.R_KNEE, Joint.R_ANKLE, -HIP_HALF_WIDTH),
    ):
        joints[:, knee] = [x, -0.05, THIGH]
        joints[:, ankle] = [x, -0.05 - SHANK, THIGH]

    upper = np.zeros((n, N_JOINTS, 3), dtype=np.float64)
    upper[:, Joint.SPINE_MID] = [0.0, SPINE_MID_HEIGHT, 0.0]
    upper[:, Joint.NECK] = [0.0, NECK_HEIGHT, 0.0]
    upper[:, Joint.HEAD] = [0.0, HEAD_HEIGHT, 0.0]
    upper[:, Joint.L_SHOULDER] = [SHOULDER_HALF_WIDTH, NECK_HEIGHT, 0.0]
    upper[:, Joint.R_SHOULDER] = [-SHOULDER_HALF_WIDTH, NECK_HEIGHT, 0.0]

    work_shoulder = Joint.L_SHOULDER if side is Side.LEFT else Joint.R_SHOULDER
    work_elbow = Joint.L_ELBOW if side is Side.LEFT else Joint.R_ELBOW
    work_wrist = Joint.L_WRIST if side is Side.LEFT else Joint.R_WRIST
    work_hand = Joint.L_HAND if side is Side.LEFT else Joint.R_HAND
    rest_elbow = Joint.R_ELBOW if side is Side.LEFT else Joint.L_ELBOW
    rest_wrist = Joint.R_WRIST if side is Side.LEFT else Joint.L_WRIST
    rest_hand = Joint.R_HAND if side is Side.LEFT else Joint.L_HAND
    rest_shoulder = Joint.R_SHOULDER if side is Side.LEFT else Joint.L_SHOULDER

    obligatory = SCAPULOHUMERAL_RISE_PER_DEG * np.maximum(
        0.0, elevation - SCAPULOHUMERAL_ONSET_DEG
    )
    upper[:, work_shoulder, 1] += (hike + obligatory) * TORSO

    theta = np.radians(elevation)
    phi = np.radians(plane)
    # Humerus direction: rotate "down" by `elevation` toward the in-plane direction selected by `plane` (0 = anterior, 90 = lateral)
    direction = np.stack(
        [
            np.sin(theta) * np.sin(phi) * sign,
            -np.cos(theta),
            np.sin(theta) * np.cos(phi),
        ],
        axis=-1,
    )
    shoulder_pos = upper[:, work_shoulder, :]
    elbow_pos = shoulder_pos + HUMERUS * direction

    # Forearm: rotate the humerus direction by the elbow angle about the humeral transverse axis, which itself is spun by the shoulder rotation angle
    anterior = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    e1 = anterior - np.sum(anterior * direction, axis=-1, keepdims=True) * direction
    e1_norm = np.linalg.norm(e1, axis=-1, keepdims=True)
    # Degenerate when the humerus is parallel to the anterior axis (arm straight forward); fall back to the lateral axis, which is always perpendicular there
    fallback = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    e1 = np.where(e1_norm > 1e-6, e1 / np.maximum(e1_norm, 1e-12), fallback)
    e2 = np.cross(direction, e1)

    psi = np.radians(rotation) * sign
    flex_axis = np.cos(psi)[:, None] * e2 + np.sin(psi)[:, None] * e1
    beta = np.radians(elbow)
    cos_b, sin_b = np.cos(beta)[:, None], np.sin(beta)[:, None]
    forearm_dir = (
        cos_b * direction
        + sin_b * np.cross(flex_axis, direction)
        + (1.0 - cos_b) * np.sum(flex_axis * direction, axis=-1, keepdims=True) * flex_axis
    )
    wrist_pos = elbow_pos + FOREARM * forearm_dir
    hand_pos = wrist_pos + HAND * forearm_dir

    upper[:, work_elbow] = elbow_pos
    upper[:, work_wrist] = wrist_pos
    upper[:, work_hand] = hand_pos

    # Resting arm hangs at the side
    rest_x = SHOULDER_HALF_WIDTH * (-sign)
    upper[:, rest_elbow] = [rest_x, NECK_HEIGHT - HUMERUS, 0.0]
    upper[:, rest_wrist] = [rest_x, NECK_HEIGHT - HUMERUS - FOREARM, 0.0]
    upper[:, rest_hand] = [rest_x, NECK_HEIGHT - HUMERUS - FOREARM - HAND, 0.0]
    upper[:, rest_shoulder] = [rest_x, NECK_HEIGHT, 0.0]

    # Apply trunk posture, then whole-body yaw
    lean_rot = _rotation(np.array([0.0, 0.0, 1.0]), lean * sign)
    flex_rot = _rotation(np.array([1.0, 0.0, 0.0]), flex)
    twist_rot = _rotation(np.array([0.0, 1.0, 0.0]), twist)
    trunk_rot = lean_rot @ flex_rot @ twist_rot

    upper_indices = [
        Joint.SPINE_MID, Joint.NECK, Joint.HEAD,
        Joint.L_SHOULDER, Joint.L_ELBOW, Joint.L_WRIST, Joint.L_HAND,
        Joint.R_SHOULDER, Joint.R_ELBOW, Joint.R_WRIST, Joint.R_HAND,
    ]
    idx = [int(j) for j in upper_indices]
    joints[:, idx, :] = np.einsum("tij,tkj->tki", trunk_rot, upper[:, idx, :])

    if abs(subject_yaw_deg) > 1e-9:
        yaw = _rotation(np.array([0.0, 1.0, 0.0]), np.full(n, subject_yaw_deg))
        joints = np.einsum("tij,tkj->tki", yaw, joints)

    return joints


def generate(
    params: MovementParams, *, n_reps: int = 1, rest_s: float = 0.0
) -> PoseSequence:
    """Generating ``n_reps`` synthetic repetitions as one continuous `PoseSequence`"""
    profile = repetition_profile(
        params.fps,
        params.concentric_s,
        params.hold_s,
        params.eccentric_s,
        n_submovements=params.n_submovements,
        return_fraction=params.return_fraction,
        n_reps=n_reps,
        rest_s=rest_s,
    )
    drive = _drive_signals(params, profile)

    joints = build_skeleton(
        elevation=drive.elevation,
        plane=drive.plane + profile * params.plane_drift_deg,
        elbow=drive.elbow + profile * params.elbow_synergy_deg,
        rotation=drive.rotation,
        side=params.side,
        trunk_lean=profile * params.trunk_lean_deg,
        trunk_flexion=profile * params.trunk_flexion_deg,
        trunk_rotation=profile * params.trunk_rotation_deg,
        shoulder_hike=profile * params.shoulder_hike_tl,
        subject_yaw_deg=params.subject_yaw_deg,
    )

    if params.noise_m > 0:
        rng = np.random.default_rng(params.seed)
        joints = joints + rng.normal(0.0, params.noise_m, size=joints.shape)

    return PoseSequence(
        joints=joints.astype(np.float32),
        fps=params.fps,
        meta={
            "synthetic": True,
            "exercise_id": params.exercise.id,
            "side": int(params.side),
            "peak_deg": params.peak_deg,
            "injected_errors": sorted(e.name for e in params.injected_errors()),
            "n_reps": int(n_reps),
        },
    )


def generate_set(params: MovementParams, n_reps: int, *, rest_s: float = 0.6) -> PoseSequence:
    """``n_reps`` repetitions with a short rest between them, as one sequence"""
    return generate(params, n_reps=n_reps, rest_s=rest_s)


@dataclass
class SyntheticCorpus:
    """TEST - A small labelled corpus for pipeline tests"""

    n_subjects: int = 8
    reps_per_condition: int = 3
    fps: float = 30.0
    seed: int = 0
    exercises: tuple[ExerciseDefinition, ...] = field(default_factory=tuple)

    def build(self) -> list[tuple[PoseSequence, MovementParams, str]]:
        """Returning ``(sequence, params, subject_id)`` triples"""
        

        exercises = self.exercises or PROTOCOL
        rng = np.random.default_rng(self.seed)
        out: list[tuple[PoseSequence, MovementParams, str]] = []

        for subject in range(self.n_subjects):
            subject_id = f"synthetic:S{subject:02d}"
            capability = float(rng.uniform(0.55, 1.05))
            lean_tendency = float(rng.uniform(0.0, 1.0))

            for exercise in exercises:
                yaw = 90.0 if exercise.required_view.value == "sagittal" else 0.0
                base_peak = 90.0 if exercise.primary_channel != "elbow_flexion" else 110.0

                for condition in range(self.reps_per_condition):
                    faulty = condition > 0
                    params = MovementParams(
                        exercise=exercise,
                        side=Side.RIGHT if subject % 2 else Side.LEFT,
                        peak_deg=base_peak * capability * (0.7 if faulty else 1.0),
                        concentric_s=exercise.tempo.concentric_s,
                        hold_s=exercise.tempo.hold_s,
                        eccentric_s=exercise.tempo.eccentric_s,
                        fps=self.fps,
                        subject_yaw_deg=yaw,
                        trunk_lean_deg=14.0 * lean_tendency if faulty else 0.0,
                        shoulder_hike_tl=0.10 if faulty else 0.0,
                        n_submovements=3 if faulty else 1,
                        noise_m=0.004,
                        seed=int(rng.integers(0, 2**31)),
                    )
                    out.append((generate(params), params, subject_id))
        return out
