# Python/ML project - folder map

Python side of the system: pose → angles → repetitions → rules → score, plus the
training and analysis code that validates it.

## Inside `rehab_ml/`

Modules at the root are the shared vocabulary every subpackage depends on.

| File | Role |
|---|---|
| `schema.py` | `PoseSequence`, the canonical 18-joint skeleton, `Joint`, `Side` |
| `taxonomy.py` | The 11 error types, severities, clinician grades |
| `adaptation.py` | Progression rules - when to advance, hold, or regress a dose |
| `cli.py` | Typer command line: train, calibrate, validate, serve |

Subpackages in pipeline order:

| Package | Role |
|---|---|
| `pose/` | Landmark extraction (`mediapipe_estimator`), gap filling and scaling (`normalize`), camera-framing checks (`view`) |
| `features/` | Joint angles (`angles`), vector maths (`geometry`), velocity and smoothness (`kinematics`), model inputs (`extractor`) |
| `segmentation/` | Splitting a continuous recording into individual repetitions |
| `exercises/` | The clinical protocol - six exercises, their targets, tempos and watched errors |
| `rules/` | Interpretable scoring: per-error criteria, the rule engine, EQI, and threshold calibration |
| `models/` | PyTorch architectures - LSTM, TCN, and the clinical hybrid, with their prediction heads |
| `data/` | Corpus adapters (`adapters/`), the repetition dataset, and a synthetic fixture |
| `training/` | Training loop plus the analyses |
| `api/` | FastAPI service the phone talks to, its request schemas, and the coach |
