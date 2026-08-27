from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import urllib.request

import numpy as np
import mediapipe as mp
import cv2

from ..schema import Joint, N_JOINTS, PoseSequence

# MediaPipe Pose landmark indices we consume
MP_NOSE = 0
MP_L_SHOULDER, MP_R_SHOULDER = 11, 12
MP_L_ELBOW, MP_R_ELBOW = 13, 14
MP_L_WRIST, MP_R_WRIST = 15, 16
MP_L_PINKY, MP_R_PINKY = 17, 18
MP_L_INDEX, MP_R_INDEX = 19, 20
MP_L_HIP, MP_R_HIP = 23, 24
MP_L_KNEE, MP_R_KNEE = 25, 26
MP_L_ANKLE, MP_R_ANKLE = 27, 28

# Canonical joints that map 1:1 onto a MediaPipe landmark
_DIRECT: dict[Joint, int] = {
    Joint.HEAD: MP_NOSE,
    Joint.L_SHOULDER: MP_L_SHOULDER,
    Joint.R_SHOULDER: MP_R_SHOULDER,
    Joint.L_ELBOW: MP_L_ELBOW,
    Joint.R_ELBOW: MP_R_ELBOW,
    Joint.L_WRIST: MP_L_WRIST,
    Joint.R_WRIST: MP_R_WRIST,
    Joint.L_HIP: MP_L_HIP,
    Joint.R_HIP: MP_R_HIP,
    Joint.L_KNEE: MP_L_KNEE,
    Joint.R_KNEE: MP_R_KNEE,
    Joint.L_ANKLE: MP_L_ANKLE,
    Joint.R_ANKLE: MP_R_ANKLE,
}

POSE_MODEL_URLS: dict[str, str] = {
    variant: (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
    )
    for variant in ("lite", "full", "heavy")
}

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH_ENV = "REHAB_ML_POSE_MODEL"


class MediaPipeUnavailable(RuntimeError):
    """Raised when the vision extras, or the model asset, are not available"""


def default_model_path(variant: str = "full") -> Path:
    return MODEL_DIR / f"pose_landmarker_{variant}.task"


def download_pose_model(variant: str = "full", dest: str | Path | None = None) -> Path:
    if variant not in POSE_MODEL_URLS:
        raise ValueError(f"unknown variant {variant!r}; choose one of {sorted(POSE_MODEL_URLS)}")

    target = Path(dest) if dest is not None else default_model_path(variant)
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(POSE_MODEL_URLS[variant], target)
    return target


def resolve_model_path(variant: str = "full", explicit: str | Path | None = None) -> Path:
    """Locate the model bundle"""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise MediaPipeUnavailable(f"pose model not found at {path}")
        return path

    from_env = os.environ.get(MODEL_PATH_ENV)
    if from_env:
        path = Path(from_env)
        if not path.is_file():
            raise MediaPipeUnavailable(f"{MODEL_PATH_ENV} points at a missing file: {path}")
        return path

    path = default_model_path(variant)
    if path.is_file():
        return path
    raise MediaPipeUnavailable(
        f"pose model {path.name} not found in {MODEL_DIR}. Fetch it with download_pose_model({variant!r}), "
        f"or point {MODEL_PATH_ENV} at an existing bundle"
    )


def landmark_confidences(landmarks) -> tuple[np.ndarray, str]:
    """Per-landmark confidence, plus which field it came from"""
    visibility = np.array([float(lm.visibility or 0.0) for lm in landmarks], dtype=np.float64)
    if float(visibility.max(initial=0.0)) > 0.0:
        return visibility, "visibility"
    presence = np.array([float(lm.presence or 0.0) for lm in landmarks], dtype=np.float64)
    if float(presence.max(initial=0.0)) > 0.0:
        return presence, "presence"
    return np.ones(len(landmarks), dtype=np.float64), "unavailable"


def landmarks_to_canonical(
    xyz: np.ndarray, visibility: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Converting one frame of 33 MediaPipe landmarks to the canonical 18 joints"""
    xyz = np.asarray(xyz, dtype=np.float64)
    vis = np.asarray(visibility, dtype=np.float64)
    if xyz.shape != (33, 3):
        raise ValueError(f"expected (33, 3) landmarks, got {xyz.shape}")

    # Axis remap: image-right -> subject-left, flip y up, flip z toward camera
    pts = np.stack([xyz[:, 0], -xyz[:, 1], -xyz[:, 2]], axis=-1)

    out = np.zeros((N_JOINTS, 3), dtype=np.float64)
    conf = np.zeros(N_JOINTS, dtype=np.float64)

    for joint, idx in _DIRECT.items():
        out[joint] = pts[idx]
        conf[joint] = vis[idx]

    pelvis = 0.5 * (pts[MP_L_HIP] + pts[MP_R_HIP])
    neck = 0.5 * (pts[MP_L_SHOULDER] + pts[MP_R_SHOULDER])
    out[Joint.PELVIS] = pelvis
    conf[Joint.PELVIS] = min(vis[MP_L_HIP], vis[MP_R_HIP])
    out[Joint.NECK] = neck
    conf[Joint.NECK] = min(vis[MP_L_SHOULDER], vis[MP_R_SHOULDER])
    out[Joint.SPINE_MID] = 0.5 * (pelvis + neck)
    conf[Joint.SPINE_MID] = min(conf[Joint.PELVIS], conf[Joint.NECK])

    # Hand centre from the index and pinky MCP knuckles
    out[Joint.L_HAND] = 0.5 * (pts[MP_L_INDEX] + pts[MP_L_PINKY])
    conf[Joint.L_HAND] = min(vis[MP_L_INDEX], vis[MP_L_PINKY])
    out[Joint.R_HAND] = 0.5 * (pts[MP_R_INDEX] + pts[MP_R_PINKY])
    conf[Joint.R_HAND] = min(vis[MP_R_INDEX], vis[MP_R_PINKY])

    return out, conf


@dataclass
class PoseEstimatorConfig:
    """MediaPipe Tasks runtime settings"""
    model: str = "full"
    model_asset_path: str | Path | None = None
    running_mode: str = "video"
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    num_poses: int = 1


class MediaPipePoseEstimator:
    """Stateful wrapper around ``PoseLandmarker``"""

    def __init__(self, config: PoseEstimatorConfig | None = None) -> None:
        self.config = config or PoseEstimatorConfig()

        mode = self.config.running_mode.lower()
        if mode not in ("image", "video"):
            raise ValueError(
                f"running_mode={self.config.running_mode!r} is not supported; "
                'use "video" (files and synchronous camera loops) or "image" (stills). '
                "See PoseEstimatorConfig for why LIVE_STREAM is excluded."
            )
        self._mode = mode
        self._model_path = resolve_model_path(self.config.model, self.config.model_asset_path)

        vision = mp.tasks.vision
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=(
                vision.RunningMode.IMAGE if mode == "image" else vision.RunningMode.VIDEO
            ),
            num_poses=self.config.num_poses,
            min_pose_detection_confidence=self.config.min_pose_detection_confidence,
            min_pose_presence_confidence=self.config.min_pose_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._mp = mp
        self._next_timestamp_ms = 0
        # Which landmark field the last detection's confidences came from
        self.confidence_source: str | None = None

    @property
    def model_path(self) -> Path:
        return self._model_path

    def __enter__(self) -> "MediaPipePoseEstimator":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self._landmarker.close()

    def process_rgb(
        self, frame_rgb: np.ndarray, timestamp_ms: int | None = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Estimating one RGB frame. Returns ``None`` when no person is detected"""
        frame = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame)

        if self._mode == "image":
            result = self._landmarker.detect(image)
        else:
            stamp = self._next_timestamp_ms if timestamp_ms is None else int(timestamp_ms)
            stamp = max(stamp, self._next_timestamp_ms)
            result = self._landmarker.detect_for_video(image, stamp)
            self._next_timestamp_ms = stamp + 1

        world = getattr(result, "pose_world_landmarks", None)
        if not world:
            return None
        landmarks = world[0]
        xyz = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
        confidence, source = landmark_confidences(landmarks)
        self.confidence_source = source
        return landmarks_to_canonical(xyz, confidence)


def _open_video(path: str | Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    return cv2, capture


def iter_video_poses(
    path: str | Path, config: PoseEstimatorConfig | None = None
) -> Iterator[tuple[int, np.ndarray | None, np.ndarray | None]]:
    """Yielding ``(frame_index, joints, confidence)`` for each frame of a video"""
    cv2, capture = _open_video(path)
    declared_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = declared_fps if declared_fps > 1.0 else 30.0
    estimator = MediaPipePoseEstimator(config)
    try:
        index = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = estimator.process_rgb(rgb, timestamp_ms=int(round(index * 1000.0 / fps)))
            if result is None:
                yield index, None, None
            else:
                yield index, result[0], result[1]
            index += 1
    finally:
        estimator.close()
        capture.release()


def sequence_from_video(
    path: str | Path,
    *,
    config: PoseEstimatorConfig | None = None,
    fps_override: float | None = None,
    max_missing_ratio: float = 0.25,
) -> PoseSequence:
    """Running pose estimation over a whole video and return a `PoseSequence`"""
    cv2, capture = _open_video(path)
    declared_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()

    fps = fps_override or (declared_fps if declared_fps > 1.0 else 30.0)

    frames: list[np.ndarray] = []
    confs: list[np.ndarray] = []
    missing = 0
    estimator_config = config or PoseEstimatorConfig()

    for _, joints, conf in iter_video_poses(path, estimator_config):
        if joints is None:
            frames.append(np.zeros((N_JOINTS, 3), dtype=np.float64))
            confs.append(np.zeros(N_JOINTS, dtype=np.float64))
            missing += 1
        else:
            frames.append(joints)
            confs.append(conf)

    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    ratio = missing / len(frames)
    if ratio > max_missing_ratio:
        raise ValueError(
            f"person not detected in {ratio:.0%} of frames (limit {max_missing_ratio:.0%}); check framing, lighting, and that the whole body is in shot"
        )

    stacked_conf = np.stack(confs)

    detected = stacked_conf.sum(axis=1) > 0
    has_confidence = bool(detected.any() and not np.allclose(stacked_conf[detected], 1.0))

    return PoseSequence(
        joints=np.stack(frames),
        fps=fps,
        confidence=stacked_conf,
        meta={
            "source": "mediapipe",
            "mediapipe_model": str(estimator_config.model),
            "video_path": str(path),
            "missing_frame_ratio": ratio,
            "declared_fps": declared_fps,
            "per_joint_confidence": has_confidence,
        },
    )
