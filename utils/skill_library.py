import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any


MANAGED_SKILLS_DIRNAME = "skills_library"


def get_managed_skills_dir(work_dir: Path) -> Path:
    skill_dir = Path(work_dir) / MANAGED_SKILLS_DIRNAME
    skill_dir.mkdir(parents=True, exist_ok=True)
    return skill_dir


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_frontmatter(skill_md_text: str) -> dict[str, str]:
    if not skill_md_text.startswith("---"):
        return {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", skill_md_text, flags=re.DOTALL)
    if not match:
        return {}
    data = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def _extract_description(skill_md_text: str) -> str:
    frontmatter = _extract_frontmatter(skill_md_text)
    if frontmatter.get("description"):
        return frontmatter["description"]
    for line in skill_md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            continue
        if stripped:
            return stripped[:240]
    return ""


def _find_skill_dirs(root_dir: Path) -> list[Path]:
    skill_dirs = []
    for skill_md in root_dir.rglob("SKILL.md"):
        skill_dir = skill_md.parent
        if any(parent in skill_dirs for parent in skill_dir.parents):
            continue
        skill_dirs.append(skill_dir)
    return sorted(skill_dirs)


def list_installed_skills(work_dir: Path) -> list[dict[str, Any]]:
    root = get_managed_skills_dir(work_dir)
    skills = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.exists():
            continue
        text = _read_text(skill_md)
        skills.append(
            {
                "name": child.name,
                "path": str(child),
                "skill_md_path": str(skill_md),
                "description": _extract_description(text),
                "preview": text[:12000],
            }
        )
    return skills


def _copy_skill_dir(src_dir: Path, work_dir: Path, overwrite: bool = True) -> dict[str, Any]:
    src_dir = Path(src_dir)
    if not (src_dir / "SKILL.md").exists():
        raise ValueError(f"'{src_dir}' is not a valid skill directory because SKILL.md is missing.")

    dest_root = get_managed_skills_dir(work_dir)
    dest_dir = dest_root / src_dir.name
    if dest_dir.exists():
        if not overwrite:
            return {
                "name": src_dir.name,
                "status": "skipped",
                "path": str(dest_dir),
                "reason": "already_exists",
            }
        shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir)
    return {
        "name": src_dir.name,
        "status": "imported",
        "path": str(dest_dir),
    }


def _safe_extract_zip(zip_path: Path, extract_dir: Path):
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target_path = extract_dir / member.filename
            target_path_resolved = target_path.resolve()
            if not str(target_path_resolved).startswith(str(extract_dir.resolve())):
                raise ValueError(f"Unsafe path detected in archive: {member.filename}")
        zf.extractall(extract_dir)


def import_skill_archive(zip_path: Path, work_dir: Path, overwrite: bool = True) -> list[dict[str, Any]]:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    with tempfile.TemporaryDirectory(prefix="paperbanana_skill_") as tmp_dir:
        extract_dir = Path(tmp_dir)
        _safe_extract_zip(zip_path, extract_dir)
        skill_dirs = _find_skill_dirs(extract_dir)
        if not skill_dirs:
            raise ValueError(f"No valid skill with SKILL.md was found in archive: {zip_path.name}")
        return [_copy_skill_dir(skill_dir, work_dir, overwrite=overwrite) for skill_dir in skill_dirs]


def import_skills_from_path(source_path: Path | str, work_dir: Path, overwrite: bool = True) -> list[dict[str, Any]]:
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Path not found: {source_path}")

    imported: list[dict[str, Any]] = []
    if source_path.is_file():
        if source_path.suffix.lower() != ".zip":
            raise ValueError("Only .zip archives are supported when importing from a file path.")
        return import_skill_archive(source_path, work_dir, overwrite=overwrite)

    if (source_path / "SKILL.md").exists():
        return [_copy_skill_dir(source_path, work_dir, overwrite=overwrite)]

    zip_files = sorted(source_path.glob("*.zip"))
    if zip_files:
        for zip_file in zip_files:
            imported.extend(import_skill_archive(zip_file, work_dir, overwrite=overwrite))
        return imported

    skill_dirs = _find_skill_dirs(source_path)
    if skill_dirs:
        for skill_dir in skill_dirs:
            imported.append(_copy_skill_dir(skill_dir, work_dir, overwrite=overwrite))
        return imported

    raise ValueError(f"No importable skills were found under: {source_path}")


def import_uploaded_archives(uploaded_archives, work_dir: Path, overwrite: bool = True) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="paperbanana_skill_upload_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for uploaded_file in uploaded_archives:
            archive_path = tmp_root / uploaded_file.name
            archive_path.write_bytes(uploaded_file.getbuffer())
            results.extend(import_skill_archive(archive_path, work_dir, overwrite=overwrite))
    return results


def delete_installed_skills(work_dir: Path, skill_names: list[str]) -> list[dict[str, Any]]:
    root = get_managed_skills_dir(work_dir)
    deleted = []
    for skill_name in skill_names:
        target = root / skill_name
        if target.exists():
            shutil.rmtree(target)
            deleted.append({"name": skill_name, "status": "deleted"})
        else:
            deleted.append({"name": skill_name, "status": "missing"})
    return deleted
