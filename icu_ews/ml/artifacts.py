"""Bundle persistence: parquet tables + joblib model artifacts + JSON metadata."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import BUNDLE_DIR

FRAME_FILES = {
    "patients": "patients.parquet",
    "observations": "observations.parquet",
    "labs": "labs.parquet",
}


def bundle_path(slug: str) -> Path:
    return BUNDLE_DIR / slug


def list_bundles() -> list[dict]:
    bundles: list[dict] = []
    if not BUNDLE_DIR.exists():
        return bundles
    for path in sorted(BUNDLE_DIR.iterdir()):
        meta_file = path / "bundle.json"
        if not path.is_dir() or not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        meta.setdefault("slug", path.name)
        bundles.append(meta)
    return bundles


def read_meta(slug: str) -> dict:
    meta_file = bundle_path(slug) / "bundle.json"
    meta = json.loads(meta_file.read_text())
    meta.setdefault("slug", slug)
    return meta


def save_bundle(slug: str, frames: dict[str, pd.DataFrame], meta: dict, model_bundle: dict) -> Path:
    path = bundle_path(slug)
    path.mkdir(parents=True, exist_ok=True)
    for name, filename in FRAME_FILES.items():
        frames[name].to_parquet(path / filename, index=False)
    joblib.dump(model_bundle, path / "bundle.joblib", compress=3)
    payload = {
        **meta,
        "metrics": model_bundle.get("metrics", {}),
        "cv": model_bundle.get("cv", {}),
        "thresholds": model_bundle.get("thresholds", {}),
        "training": model_bundle.get("training", {}),
    }
    (path / "bundle.json").write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_frames(slug: str) -> dict[str, pd.DataFrame]:
    path = bundle_path(slug)
    frames = {name: pd.read_parquet(path / filename) for name, filename in FRAME_FILES.items()}
    frames["patients"]["admit_ts"] = pd.to_datetime(frames["patients"]["admit_ts"])
    frames["patients"]["snapshot_ts"] = pd.to_datetime(frames["patients"]["snapshot_ts"])
    frames["observations"]["ts"] = pd.to_datetime(frames["observations"]["ts"])
    frames["labs"]["ts"] = pd.to_datetime(frames["labs"]["ts"])
    return frames


def _refresh_preprocessing(model_bundle: dict) -> None:
    """Rebuild imputers/scalers with the installed scikit-learn version.

    Pickled preprocessing steps from an older scikit-learn can lack private
    attributes that newer releases read in ``transform`` (e.g. SimpleImputer's
    ``_fill_dtype``). Their fitted state is fully determined by ``X_train``, so
    refitting fresh steps reproduces the same statistics while the trained
    estimators are kept exactly as saved.
    """
    X_train = model_bundle.get("X_train")
    if X_train is None:
        return
    for name, model in model_bundle.get("models", {}).items():
        if not isinstance(model, Pipeline):
            continue
        *preprocess_steps, final_step = model.steps
        data = X_train
        rebuilt: list[tuple[str, object]] = []
        for step_name, step in preprocess_steps:
            if isinstance(step, SimpleImputer):
                fresh = SimpleImputer(**step.get_params(deep=False))
            elif isinstance(step, StandardScaler):
                fresh = StandardScaler(**step.get_params(deep=False))
            else:
                rebuilt.append((step_name, step))
                data = step.transform(data)
                continue
            fresh.fit(data)
            data = fresh.transform(data)
            rebuilt.append((step_name, fresh))
        model_bundle["models"][name] = Pipeline([*rebuilt, final_step])


def load_model_bundle(slug: str) -> dict:
    bundle = joblib.load(bundle_path(slug) / "bundle.joblib")
    _refresh_preprocessing(bundle)
    return bundle


def bundle_exists(slug: str) -> bool:
    return (bundle_path(slug) / "bundle.joblib").exists()


def default_slug() -> str | None:
    bundles = list_bundles()
    if not bundles:
        return None
    demo = [b for b in bundles if b.get("is_demo")]
    return (demo[0] if demo else bundles[0])["slug"]
