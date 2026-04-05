from __future__ import annotations

import json
import logging
import socket
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

from configs import PathConfig
from paths import job_dir
from utils import atomic_write_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def read_yaml_or_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must parse to a mapping.")
    return data


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_job_dirs(job_path: Path) -> None:
    for d in [
        job_path,
        job_path / "logs",
        job_path / "checkpoints",
        job_path / "artifacts",
        job_path / "artifacts" / "plots",
        job_path / "artifacts" / "ftle_chunks",
        job_path / "artifacts" / "margin_chunks",
    ]:
        d.mkdir(parents=True, exist_ok=True)


def default_status(job_id: str) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "train": {"state": "pending"},
        "eval": {"state": "pending"},
        "plots": {"state": "pending"},
        "updated_at": utc_now_iso(),
    }


def status_path(job_path: Path) -> Path:
    return job_path / "status.json"


def spec_path(job_path: Path) -> Path:
    return job_path / "spec.json"


def load_status(job_path: Path, job_id: str) -> Dict[str, Any]:
    return read_json(status_path(job_path), default_status(job_id))


def save_status(job_path: Path, status: Dict[str, Any]) -> None:
    status["updated_at"] = utc_now_iso()
    write_json(status_path(job_path), status)


def write_spec_once(job_path: Path, spec: Dict[str, Any]) -> None:
    path = spec_path(job_path)
    normalized = _normalized_job_spec(spec)
    if path.exists():
        existing = read_json(path)
        existing_normalized = _normalized_job_spec(existing)
        if existing_normalized != normalized:
            raise ValueError(f"Spec mismatch for existing job folder: {job_path}")
        merged = dict(existing)
        merged["experiment_names"] = sorted(
            set(existing.get("experiment_names", _spec_experiment_names(existing)))
            | set(_spec_experiment_names(spec))
        )
        write_json(path, merged)
        return
    payload = dict(spec)
    payload["experiment_names"] = sorted(_spec_experiment_names(spec))
    write_json(path, payload)


def stage_logger(job_path: Path, stage: str) -> logging.Logger:
    logger_name = f"mnist_ftle.{job_path.name}.{stage}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_path = job_path / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(f"[{job_path.name}:{stage}] %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def log_stage_header(logger: logging.Logger, spec: Dict[str, Any]) -> None:
    logger.info("host=%s", socket.gethostname())
    logger.info("job_id=%s", spec["job_id"])
    logger.info("dataset=%s", spec.get("dataset", "mnist"))
    logger.info("spec=%s", json.dumps(spec, sort_keys=True))


def record_failure(job_path: Path, stage: str, status: Dict[str, Any], exc: BaseException) -> None:
    tb_path = job_path / "logs" / f"{stage}_traceback.log"
    with open(tb_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{utc_now_iso()}]\n")
        f.write(traceback.format_exc())
        f.write("\n")
    stage_status = status.setdefault(stage, {})
    stage_status["state"] = "failed"
    stage_status["error"] = str(exc)
    stage_status["traceback_file"] = str(tb_path.relative_to(job_path))
    save_status(job_path, status)


def resolve_job_path(paths: PathConfig, spec: Dict[str, Any]) -> Path:
    return job_dir(paths, spec.get("dataset", "mnist"), spec["job_id"])


def _normalized_job_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": spec.get("job_id"),
        "dataset": spec.get("dataset", "mnist"),
        "train": spec.get("train", {}),
        "eval": spec.get("eval", {}),
        "plots": spec.get("plots", {}),
    }


def _spec_experiment_names(spec: Dict[str, Any]) -> list[str]:
    names = spec.get("experiment_names")
    if isinstance(names, list):
        return [str(name) for name in names]
    name = spec.get("experiment_name")
    if name is None:
        return []
    return [str(name)]
