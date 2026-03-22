import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_ROOT = PROJECT_ROOT / "build" / "portable_build"
APP_TEMPLATE_DIR = BUILD_ROOT / "app_template"
DIST_DIR = PROJECT_ROOT / "dist"

INCLUDE_DIRS = [
    "agents",
    "assets",
    "configs",
    "data",
    "notebooks",
    "prompts",
    "skill",
    "skills_library",
    "static",
    "style_guides",
    "utils",
    "visualize",
]

INCLUDE_FILES = [
    "demo.py",
    "README.md",
    "LICENSE",
    "requirements.txt",
]

EXCLUDED_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "logs",
    "results",
    "dist",
    "build",
    ".playwright-cli",
    "configs/model_config.yaml",
}

COLLECT_ALL_PACKAGES = [
    "streamlit",
    "google.genai",
    "google.auth",
    "anthropic",
    "openai",
    "aiofiles",
    "yaml",
    "json_repair",
    "matplotlib",
    "numpy",
    "tqdm",
    "dotenv",
    "huggingface_hub",
    "PIL",
]


def should_exclude(path: Path) -> bool:
    normalized = path.as_posix()
    if normalized in EXCLUDED_NAMES:
        return True
    return any(part in EXCLUDED_NAMES for part in path.parts)


def as_windows_long_path(path: Path) -> str:
    raw = str(path.resolve())
    if os.name != "nt":
        return raw
    if raw.startswith("\\\\?\\"):
        return raw
    return "\\\\?\\" + raw


def copy_path(src: Path, dst: Path):
    if src.is_dir():
        for child in src.rglob("*"):
            relative = child.relative_to(src)
            if should_exclude(relative):
                continue
            target = dst / relative
            if child.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(as_windows_long_path(child), as_windows_long_path(target))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(as_windows_long_path(src), as_windows_long_path(dst))


def prepare_app_template():
    if BUILD_ROOT.exists():
        shutil.rmtree(as_windows_long_path(BUILD_ROOT), ignore_errors=True)
    APP_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    for relative_dir in INCLUDE_DIRS:
        src = PROJECT_ROOT / relative_dir
        if src.exists():
            copy_path(src, APP_TEMPLATE_DIR / relative_dir)

    for relative_file in INCLUDE_FILES:
        src = PROJECT_ROOT / relative_file
        if src.exists():
            copy_path(src, APP_TEMPLATE_DIR / relative_file)

    template_config = PROJECT_ROOT / "configs" / "model_config.template.yaml"
    runtime_config = APP_TEMPLATE_DIR / "configs" / "model_config.yaml"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_config, runtime_config)

    bundle_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    (APP_TEMPLATE_DIR / ".bundle_version").write_text(bundle_version, encoding="utf-8")
    return bundle_version


def ensure_pyinstaller(python_executable: str):
    try:
        subprocess.run(
            [python_executable, "-m", "pip", "--version"],
            cwd=str(PROJECT_ROOT),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            [python_executable, "-m", "ensurepip", "--upgrade"],
            cwd=str(PROJECT_ROOT),
            check=True,
        )

    subprocess.run(
        [python_executable, "-m", "pip", "install", "pyinstaller"],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def build_executable(python_executable: str):
    add_data_arg = f"{APP_TEMPLATE_DIR};app_template"
    command = [
        python_executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        "PaperBanana",
        "--add-data",
        add_data_arg,
    ]
    for package in COLLECT_ALL_PACKAGES:
        command.extend(["--collect-all", package])
    command.append("portable_launcher.py")

    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def main():
    python_executable = sys.executable
    print(f"[build] Using Python: {python_executable}")
    bundle_version = prepare_app_template()
    print(f"[build] Prepared sanitized app template (version {bundle_version})")
    ensure_pyinstaller(python_executable)
    print("[build] PyInstaller is ready")
    build_executable(python_executable)
    exe_path = DIST_DIR / "PaperBanana.exe"
    print(f"[build] Portable executable created at: {exe_path}")


if __name__ == "__main__":
    main()
