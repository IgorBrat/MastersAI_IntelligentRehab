from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from ..adaptation import Action, Decision, SessionRecord
from ..exercises.base import ExerciseDefinition
from ..taxonomy import ERROR_SPECS
from ..exercises.registry import get_exercise


DEFAULT_MODEL = "gemini-2.5-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

TIMEOUT_S = 8.0

CUES_UK = {spec.name_short: spec.patient for spec in ERROR_SPECS.values()}

_ACTION_UK = {
    Action.NEED_MORE_SESSIONS: "Продовжуйте у тому ж режимі - потрібно більше занять, щоб оцінити прогрес.",
    Action.HOLD: "Залишаємо навантаження без змін: ви працюєте на потрібному рівні складності.",
    Action.SAFETY_HOLD: "Навантаження не змінюємо, поки не приберемо помилку в техніці.",
    Action.REDUCE_VOLUME: "Зменшуємо кількість повторень: якість помітно падала до кінця підходу.",
    Action.RESTORE_VOLUME: "Повертаємо звичайну кількість повторень - якість стабільна.",
    Action.REGRESS_RANGE: "Тимчасово зменшуємо цільову амплітуду, щоб рух виконувався чисто.",
    Action.ADVANCE_RANGE: "Підвищуємо цільову амплітуду - ви впевнено виконуєте поточну.",
    Action.UNLOCK_NEXT: "Ви досягли верхньої межі для цієї вправи. Час додати наступну.",
}


def top_errors(session: SessionRecord | None, limit: int = 2) -> list[tuple[str, float]]:
    """Ranking a session's errors by how often they fired, dropping the zeros"""
    if session is None or not session.error_frequency:
        return []
    ranked = sorted(session.error_frequency.items(), key=lambda kv: -kv[1])
    return [(name, rate) for name, rate in ranked if rate > 0][:limit]


def template_text(
    decision: Decision,
    exercise: ExerciseDefinition,
    last_session: SessionRecord | None = None,
) -> str:
    """Assembling deterministic Ukrainian text from the decision and the error cues"""
    parts = [_ACTION_UK.get(decision.action, "")]

    if decision.action is Action.UNLOCK_NEXT and decision.unlocked_exercise_id:
        parts.append(f"Наступна вправа: {get_exercise(decision.unlocked_exercise_id).name_uk}.")

    for slug, rate in top_errors(last_session):
        cue = CUES_UK.get(slug)
        if cue:
            parts.append(f"{cue} (у {rate:.0%} повторень)")

    return " ".join(p for p in parts if p).strip()


def _prompt(
    decision: Decision,
    exercise: ExerciseDefinition,
    last_session: SessionRecord | None,
) -> str:
    """Building the generation prompt, listing everything the model may use"""
    errors = top_errors(last_session)
    error_lines = (
        "\n".join(
            f"- {CUES_UK.get(slug, slug)} (у {rate:.0%} повторень)"
            for slug, rate in errors
        )
        or "- помилок техніки не зафіксовано"
    )
    return f"""Ти асистент реабілітолога. Напиши коротке повідомлення пацієнтові українською.

Вправа: {exercise.name_uk}
Рішення програми: {_ACTION_UK.get(decision.action, decision.action.value)}
Підстава (технічна, не переказуй дослівно): {decision.reason}
Що варто виправити:
{error_lines}

ПРАВИЛА:
- 2-3 речення, звертайся на "ви", тепло але без надмірних похвал.
- НЕ додавай жодних чисел, градусів, кількості повторень чи відсотків. Їх покаже застосунок.
- НЕ давай медичних порад, НЕ згадуй діагнози, НЕ пропонуй інших вправ.
- Якщо є помилка техніки - назви одну, найважливішу, і як її виправити.
- Тільки текст повідомлення, без заголовків і форматування."""


def _call_gemini(prompt: str, api_key: str, model: str, timeout: float) -> str:
    """Sending one generation request and returning its text"""
    request = urllib.request.Request(
        f"{ENDPOINT.format(model=model)}?key={api_key}",
        data=json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                # Gemini 2.5 "thinking" tokens count against maxOutputTokens: at
                # 500 the message was cut off after two lines. A templated
                # rewording needs no reasoning, so thinking is switched off.
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1024,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    candidate = payload["candidates"][0]
    parts = candidate["content"]["parts"]
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise ValueError("empty generation")
    # A message cut off mid-sentence is worse than the template: never return one
    finish = candidate.get("finishReason")
    if finish not in (None, "STOP"):
        raise ValueError(f"generation ended with {finish}")
    return text


def phrase(
    decision: Decision,
    exercise: ExerciseDefinition,
    last_session: SessionRecord | None = None,
    *,
    timeout: float = TIMEOUT_S,
) -> tuple[str, str, str | None]:
    """Producing the coaching text, falling back to the template on any failure

    Returns ``(text, source, error)`` where source is "gemini" or "template", so the app can show
    that the coach ran degraded rather than presenting template text as generated
    """
    fallback = template_text(decision, exercise, last_session)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return fallback, "template", "GEMINI_API_KEY not set"

    model = os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    try:
        return _call_gemini(_prompt(decision, exercise, last_session), api_key, model,
                            timeout), "gemini", None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200] if exc.fp else ""
        return fallback, "template", f"HTTP {exc.code}: {detail}"
    except Exception as exc:
        return fallback, "template", f"{type(exc).__name__}: {exc}"
