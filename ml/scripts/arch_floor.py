"""Is the model-side null architecture-specific, or does it hold across encoders?"""

from __future__ import annotations

import itertools
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rehab_ml.data.adapters.intellirehab import IntelliRehabAdapter
from rehab_ml.data.dataset import build_dataset
from rehab_ml.features.angles import AngleSeries
from rehab_ml.training.train import TrainConfig, cross_validate

OUT = Path("reports/arch_floor.json")
EXERCISE = "elbow_flexion_extension"
MODELS = ("lstm", "tcn")
ARMS = ("off", "on")
SEEDS = (0, 1, 2, 3, 4)


@contextmanager
def correction(enabled: bool):
    original = AngleSeries.shoulder_hike

    def patched(self, baseline_frames=5, *, scapular_correction=True):
        return original(self, baseline_frames, scapular_correction=enabled)

    AngleSeries.shoulder_hike = patched
    try:
        yield
    finally:
        AngleSeries.shoulder_hike = original


def run(model: str, arm: str, seed: int) -> dict:
    with correction(arm == "on"):
        dataset = build_dataset(
            IntelliRehabAdapter("../data/IntelliRehabDS"),
            mirror_augment=False,
            self_paced=True,
        ).filter(exercise_id=EXERCISE)
        print(f"{len(dataset)} repetitions, {len(dataset.subjects)} subjects")
        report = cross_validate(dataset, TrainConfig(model=model, epochs=30, seed=seed))
    patient = report.get("by_cohort", {}).get("patient", {})
    return {
        "model": model,
        "arm": arm,
        "seed": seed,
        "n": len(dataset),
        "balanced_all": report["binary_correct"]["balanced_accuracy"],
        "balanced_patients": patient.get("model", {}).get("balanced_accuracy"),
    }


def load() -> dict:
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}


def save(model: str, arm: str, seed: int, value: dict) -> None:
    store = load()
    store.setdefault(model, {})[f"{arm}#s{seed}"] = value
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(store, indent=2), encoding="utf-8")
    print(f"\nwritten to {OUT}")


def sign_flip_p(d) -> float:
    a = np.asarray(d, dtype=float)
    obs = abs(a.mean())
    g = list(itertools.product([1.0, -1.0], repeat=a.size))
    return sum(abs((a * np.array(x)).mean()) >= obs - 1e-12 for x in g) / len(g)


def hybrid_deltas() -> list[float] | None:
    """Reading the hybrid arm back from prior_vs_learned.json rather than recomputing"""
    p = Path("reports/prior_vs_learned.json")
    if not p.exists():
        return None
    pvl = json.loads(p.read_text(encoding="utf-8"))["elbow"]
    k = lambda a, s: a if s == 0 else f"{a}#s{s}"  # noqa: E731
    try:
        return [
            pvl[k("both", s)]["balanced_patients"] - pvl[k("neither", s)]["balanced_patients"]
            for s in SEEDS
        ]
    except KeyError:
        return None


def compare() -> None:
    store = load()
    rows = []
    hd = hybrid_deltas()
    if hd:
        rows.append(("hybrid (from prior_vs_learned)", hd))
    for model in MODELS:
        e = store.get(model, {})
        seeds = [s for s in SEEDS if f"off#s{s}" in e and f"on#s{s}" in e]
        if len(seeds) < 2:
            print(f"{model}: {len(seeds)} paired seed(s) - need at least 2")
            continue
        rows.append((model, [e[f"on#s{s}"]["balanced_patients"]
                             - e[f"off#s{s}"]["balanced_patients"] for s in seeds]))

    print(f"\n=== IRDS elbow, correction off vs on, patient cohort ===")
    print(f"{'architecture':32}{'mean':>10}{'sd':>9}{'sign stable':>13}{'p':>8}")
    for name, d in rows:
        a = np.array(d)
        stable = bool(np.all(a > 0) or np.all(a < 0))
        print(f"{name:32}{a.mean():>+10.4f}{a.std(ddof=1):>9.4f}{str(stable):>13}{sign_flip_p(d):>8.3f}")

    for name, d in rows:
        print(f"\n{name}: " + "  ".join(f"{v:+.4f}" for v in d))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "compare":
        compare()
        raise SystemExit(0)
    model = sys.argv[1]
    arm = sys.argv[2] if len(sys.argv) > 2 else ""
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if model not in MODELS or arm not in ARMS:
        print(__doc__)
        raise SystemExit(2)
    save(model, arm, seed, run(model, arm, seed))
