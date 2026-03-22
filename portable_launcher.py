import argparse
import ctypes
import logging
import os
import shutil
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

# Runtime dependency imports so PyInstaller collects the packages used by demo.py.
import aiofiles  # noqa: F401
import anthropic  # noqa: F401
import dotenv  # noqa: F401
import google.auth  # noqa: F401
import google.genai  # noqa: F401
import huggingface_hub  # noqa: F401
import json_repair  # noqa: F401
import matplotlib  # noqa: F401
import numpy  # noqa: F401
import openai  # noqa: F401
import PIL.Image  # noqa: F401
import streamlit.web.bootstrap as bootstrap
import tqdm  # noqa: F401
import yaml  # noqa: F401


APP_NAME = "PaperBanana"
APP_DATA_DIRNAME = "PaperBananaPortable"
BUNDLE_DIRNAME = "app_template"
PREFERRED_PORT = 8501
PORT_SCAN_LIMIT = 50
PRESERVED_PATHS = {
    Path("configs") / "model_config.yaml",
    Path("results"),
    Path("logs"),
    Path("skills_library"),
}


def get_bundle_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def get_template_root() -> Path:
    return get_bundle_root() / BUNDLE_DIRNAME


def get_default_app_dir() -> Path:
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return local_app_data / APP_DATA_DIRNAME


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(app_dir: Path) -> Path:
    logs_dir = ensure_directory(app_dir / "logs")
    log_path = logs_dir / "portable_launcher.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


def show_error_dialog(message: str):
    try:
        ctypes.windll.user32.MessageBoxW(0, message, APP_NAME, 0x10)
    except Exception:
        pass


def read_bundle_version(template_root: Path) -> str:
    version_file = template_root / ".bundle_version"
    if version_file.exists():
        return version_file.read_text(encoding="utf-8", errors="ignore").strip()
    return "dev"


def is_preserved_path(relative_path: Path) -> bool:
    for preserved in PRESERVED_PATHS:
        if relative_path == preserved or preserved in relative_path.parents:
            return True
    return False


def sync_template_to_app_dir(template_root: Path, app_dir: Path):
    bundle_version = read_bundle_version(template_root)
    version_marker = app_dir / ".bundle_version"
    current_version = version_marker.read_text(encoding="utf-8", errors="ignore").strip() if version_marker.exists() else ""

    if current_version == bundle_version and (app_dir / "demo.py").exists():
        logging.info("Portable app directory is already up to date: %s", app_dir)
        return

    logging.info("Syncing bundled app template into %s", app_dir)
    for src in template_root.rglob("*"):
        relative_path = src.relative_to(template_root)
        dest = app_dir / relative_path
        if is_preserved_path(relative_path) and dest.exists():
            continue
        if src.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    version_marker.write_text(bundle_version, encoding="utf-8")


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def choose_port(preferred_port: int) -> int:
    for port in range(preferred_port, preferred_port + PORT_SCAN_LIMIT):
        if is_port_available(port):
            return port
    raise RuntimeError("No free localhost port was found for Streamlit.")


def server_is_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


def start_browser_opener(url: str, open_browser: bool):
    def wait_for_server():
        for _ in range(240):
            if server_is_ready(url):
                logging.info("Server became ready at %s", url)
                if open_browser:
                    webbrowser.open(url)
                return
            time.sleep(0.5)
        logging.warning("Timed out while waiting for Streamlit to start at %s", url)

    threading.Thread(target=wait_for_server, daemon=True).start()


def build_flag_options(port: int) -> dict[str, object]:
    return {
        "server_headless": True,
        "server_port": port,
        "server_address": "127.0.0.1",
        "server_fileWatcherType": "none",
        "browser_gatherUsageStats": False,
        "global_developmentMode": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable launcher for PaperBanana")
    parser.add_argument("--app-dir", default="", help="Override the extracted app directory")
    parser.add_argument("--port", type=int, default=PREFERRED_PORT, help="Preferred Streamlit port")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")

    app_dir = Path(args.app_dir).resolve() if args.app_dir else get_default_app_dir()
    ensure_directory(app_dir)
    log_path = setup_logging(app_dir)
    logging.info("Launcher starting. App dir: %s", app_dir)

    try:
        template_root = get_template_root()
        if not template_root.exists():
            raise RuntimeError(f"Bundled app template was not found at {template_root}")

        sync_template_to_app_dir(template_root, app_dir)
        os.chdir(app_dir)

        port = choose_port(args.port)
        url = f"http://127.0.0.1:{port}"
        flag_options = build_flag_options(port)
        os.environ["STREAMLIT_SERVER_PORT"] = str(port)
        os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        bootstrap.load_config_options(flag_options)
        start_browser_opener(url, open_browser=not args.no_browser)
        logging.info("Launching Streamlit at %s", url)

        bootstrap.run(
            main_script_path=str(app_dir / "demo.py"),
            is_hello=False,
            args=[],
            flag_options=flag_options,
        )
    except Exception as exc:
        logging.exception("Portable launcher failed: %s", exc)
        message = f"{APP_NAME} failed to start.\n\nError: {exc}\n\nLog file: {log_path}"
        show_error_dialog(message)
        raise


if __name__ == "__main__":
    main()
