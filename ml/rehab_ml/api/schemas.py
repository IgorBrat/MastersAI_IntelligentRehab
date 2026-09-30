from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# MediaPipe Pose emits a fixed 33-landmark topology.
N_LANDMARKS = 33

MAX_FRAMES = 5400


class PoseFrames(BaseModel):
    """Carrying a window of MediaPipe world landmarks with its frame rate"""
    exercise_id: str
    side: str = Field(description="'left' or 'right'")
    fps: float = Field(gt=0.0, le=240.0)
    landmarks: list[list[list[float]]] = Field(
        description="(T, 33, 3) world landmarks in metres, MediaPipe axis convention."
    )
    visibility: list[list[float]] | None = Field(
        default=None, description="(T, 33) per-landmark visibility in [0, 1]."
    )

    @field_validator("landmarks")
    @classmethod
    def _check_shape(cls, value):
        if not value:
            raise ValueError("landmarks is empty")
        if len(value) > MAX_FRAMES:
            raise ValueError(f"{len(value)} frames exceeds the {MAX_FRAMES}-frame limit")
        for i, frame in enumerate(value):
            if len(frame) != N_LANDMARKS:
                raise ValueError(f"frame {i}: expected {N_LANDMARKS} landmarks, got {len(frame)}")
            for point in frame:
                if len(point) != 3:
                    raise ValueError(f"frame {i}: expected 3 coordinates per landmark")
        return value

    @field_validator("visibility")
    @classmethod
    def _check_visibility(cls, value):
        if value is None:
            return value
        for i, frame in enumerate(value):
            if len(frame) != N_LANDMARKS:
                raise ValueError(f"visibility frame {i}: expected {N_LANDMARKS} values")
        return value


class VideoRequest(BaseModel):
    """TEST - Naming one of the service's own video files to score"""
    exercise_id: str
    side: str = Field(description="'left' or 'right'")
    relative_path: str = Field(
        description="Path relative to the served video directory, from GET /videos."
    )


class ExerciseInfo(BaseModel):
    id: str
    name_uk: str
    name_en: str
    primary_channel: str
    required_view: str
    target_deg: float
    floor_deg: float
    ceiling_deg: float
    concentric_s: float
    hold_s: float
    eccentric_s: float
    default_reps: int
    instructions_uk: str
    watched_errors: list[dict]


class ReadinessInfo(BaseModel):
    """Whether the framing is good enough to start a set"""

    ready: bool
    view_ok: bool
    joints_ok: bool
    required_view: str
    detected_view: str
    yaw_deg: float
    message_uk: str
    missing_joints: list[str] = Field(default_factory=list)


class ServiceInfo(BaseModel):
    status: str
    self_paced: bool
    thresholds: str | None
    thresholds_provenance: dict | None

    thresholds_sha256: str | None = None
    git_sha: str | None = None
    #: Criteria switched off for this sensor, by slug. Empty is the normal case
    excluded_criteria: list[str] = Field(default_factory=list)
    model: dict | None = None
    warnings: list[str] = Field(default_factory=list)


class SessionHistoryItem(BaseModel):
    """Holding one completed session as the phone stored it.
    Aggregates are nullable because a rejected recording has no repetitions to aggregate over,
    and the coach must tolerate one in the history
    """
    exercise_id: str
    timestamp: str = Field(default="", description="ISO 8601, produced by the client.")
    mean_eqi: float | None = None
    n_reps: int = 0
    correct_ratio: float | None = None
    fatigue_trend: float = 0.0
    error_frequency: dict[str, float] = Field(default_factory=dict)


class CoachRequest(BaseModel):
    """Carrying the phone's whole history plus the dose currently in force

    The client owns its history, which keeps the service stateless and
    single-user by construction
    """

    exercise_id: str
    side: str = Field(description="'left' or 'right'")
    history: list[SessionHistoryItem] = Field(default_factory=list)
    # Omitted dose fields fall back to the protocol default for the exercise.
    rom_target: float | None = None
    reps: int | None = None
    sets: int | None = None
    unlocked: list[str] = Field(default_factory=list)


class CoachResponse(BaseModel):
    """Progression decision"""
    action: str
    reason: str
    changed: bool
    unlocked_exercise_id: str | None
    prescription: dict
    evidence: dict
    text: str
    # "gemini" or "template"
    text_source: str
    text_error: str | None = None
