"""Measuring the scapulohumeral correction on both AUC and a subject-disjoint threshold"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rehab_ml.data.adapters.intellirehab import IntelliRehabAdapter  
from rehab_ml.data.adapters.ui_prmd import UiPrmdAdapter  
from rehab_ml.data.dataset import build_dataset  
from rehab_ml.features.angles import AngleSeries  
from rehab_ml.training.eqi_validation import discriminate  


@contextmanager
def correction(enabled: bool):
    """Forcing `shoulder_hike` on or off for the duration of the block

    Must wrap `build_dataset`: the EQI components and penalties are computed at
    build time and cached in sample.meta
    """
    original = AngleSeries.shoulder_hike

    def patched(self, baseline_frames=5, *, scapular_correction=True):
        return original(
            self, baseline_frames, scapular_correction=enabled
        )

    AngleSeries.shoulder_hike = patched
    try:
        yield
    finally:
        AngleSeries.shoulder_hike = original


ARMS = [
    ("IRDS patients", lambda: IntelliRehabAdapter("../data/IntelliRehabDS"), "patient"),
    ("IRDS all", lambda: IntelliRehabAdapter("../data/IntelliRehabDS"), None),
    ("UI-PRMD vicon", lambda: UiPrmdAdapter("../data/UIPRMD", sensor="vicon"), None),
    ("UI-PRMD kinect", lambda: UiPrmdAdapter("../data/UIPRMD", sensor="kinect"), None),
]


def measure(factory, cohort: str, *, enabled: bool) -> dict:
    """Building the dataset under one correction setting and scoring EQI on it"""
    with correction(enabled):
        dataset = build_dataset(factory(), mirror_augment=False, self_paced=True)
        result = discriminate(dataset, cohort=cohort, n_resamples=2000)
    if "formulations" not in result:
        return {"error": result.get("error", "no formulations")}
    gated = result["formulations"]["gated"]
    return {
        "n": result["n"],
        "auc": gated["auc"],
        "auc_ci": gated["auc_ci"],
        "bal_loso": gated["balanced_accuracy_loso"],
        "bal_loso_ci": gated["balanced_accuracy_loso_ci"],
        "bal_median": gated["balanced_accuracy_at_median"],
    }


def main() -> None:
    """Running every arm and printing the AUC and thresholded deltas side by side"""
    out: dict = {}
    header = (
        f"{'arm':16}{'n':>6}"
        f"{'AUC off':>9}{'AUC on':>9}{'dAUC':>9}"
        f"{'bal off':>9}{'bal on':>9}{'dbal':>9}  sign"
    )
    print(header)
    print("-" * len(header))

    for label, factory, cohort in ARMS:
        off = measure(factory, cohort, enabled=False)
        on = measure(factory, cohort, enabled=True)
        if "error" in off or "error" in on:
            print(f"{label:16}  {off.get('error') or on.get('error')}")
            out[label] = {"off": off, "on": on}
            continue

        d_auc = on["auc"] - off["auc"]
        d_bal = on["bal_loso"] - off["bal_loso"]
        agree = "agree" if (d_auc > 0) == (d_bal > 0) else "DISAGREE"
        print(
            f"{label:16}{on['n']:>6}"
            f"{off['auc']:>9.4f}{on['auc']:>9.4f}{d_auc:>+9.4f}"
            f"{off['bal_loso']:>9.4f}{on['bal_loso']:>9.4f}{d_bal:>+9.4f}  {agree}"
        )
        out[label] = {
            "off": off,
            "on": on,
            "delta_auc": round(d_auc, 4),
            "delta_bal_loso": round(d_bal, 4),
            "measures_agree": agree == "agree",
        }

    resolved = [v for v in out.values() if "delta_auc" in v]
    print()
    print(f"AUC positive on {sum(v['delta_auc'] > 0 for v in resolved)} of {len(resolved)}")
    print(
        f"LOSO-thresholded positive on "
        f"{sum(v['delta_bal_loso'] > 0 for v in resolved)} of {len(resolved)}"
    )

    target = Path("reports/scapular_ab.json")
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten to {target}")


if __name__ == "__main__":
    main()
