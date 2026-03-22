import json
from datetime import datetime
from pathlib import Path
from typing import Any


def get_history_root(work_dir: Path) -> Path:
    root = Path(work_dir) / "results" / "history"
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_task_id(prefix: str = "task") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def get_task_dir(work_dir: Path, task_id: str) -> Path:
    task_dir = get_history_root(work_dir) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def build_task_record(task_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    timestamp = datetime.now().isoformat(timespec="seconds")
    return {
        "task_id": task_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "status": "running",
        "events": [],
        "results_file": "",
        "export_json_file": "",
        "error": "",
        "summary": {},
        **metadata,
    }


def _write_json(path: Path, payload: dict[str, Any] | list[Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_task_record(work_dir: Path, record: dict[str, Any]) -> Path:
    record = dict(record)
    record["updated_at"] = datetime.now().isoformat(timespec="seconds")
    record_path = get_task_dir(work_dir, record["task_id"]) / "task.json"
    _write_json(record_path, record)
    return record_path


def save_task_results(work_dir: Path, task_id: str, results: list[dict[str, Any]]) -> Path:
    results_path = get_task_dir(work_dir, task_id) / "results.json"
    _write_json(results_path, results)
    return results_path


def load_task_record(work_dir: Path, task_id: str) -> dict[str, Any] | None:
    record_path = get_task_dir(work_dir, task_id) / "task.json"
    if not record_path.exists():
        return None
    return json.loads(record_path.read_text(encoding="utf-8"))


def load_task_results(work_dir: Path, task_id: str) -> list[dict[str, Any]]:
    results_path = get_task_dir(work_dir, task_id) / "results.json"
    if not results_path.exists():
        return []
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def list_task_records(work_dir: Path, limit: int = 30) -> list[dict[str, Any]]:
    records = []
    for record_path in sorted(
        get_history_root(work_dir).glob("*/task.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            records.append(record)
        except Exception:
            continue
        if len(records) >= limit:
            break
    return records
