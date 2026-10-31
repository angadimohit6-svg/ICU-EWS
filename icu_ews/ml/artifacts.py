"""Bundle persistence: parquet tables + joblib model artifacts + JSON metadata."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

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


def load_model_bundle(slug: str) -> dict:
    return joblib.load(bundle_path(slug) / "bundle.joblib")


def bundle_exists(slug: str) -> bool:
    return (bundle_path(slug) / "bundle.joblib").exists()


def default_slug() -> str | None:
    bundles = list_bundles()
    if not bundles:
        return None
    demo = [b for b in bundles if b.get("is_demo")]
    return (demo[0] if demo else bundles[0])["slug"]
