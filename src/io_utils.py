from __future__ import annotations

import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def requests_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": "offline-reentry-risk-prototype/1.0"})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_file(url: str, target_path: Path, force: bool = False, timeout: int = 60) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not force:
        LOGGER.info("Using cached file: %s", target_path)
        return target_path

    LOGGER.info("Downloading %s", url)
    session = requests_session()
    with session.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with target_path.open("wb") as handle:
            shutil.copyfileobj(response.raw, handle)
    return target_path


def fetch_text(url: str, target_path: Path | None = None, force: bool = False, timeout: int = 60) -> str:
    if target_path and target_path.exists() and not force:
        return target_path.read_text(encoding="utf-8")

    LOGGER.info("Fetching %s", url)
    session = requests_session()
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    text = response.text
    if target_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(text, encoding="utf-8")
    return text


def unzip_archive(zip_path: Path, destination_dir: Path, force: bool = False) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for item in destination_dir.glob("*"):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination_dir)
    return destination_dir


def find_first_file(directory: Path, patterns: Iterable[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(directory.rglob(pattern))
        if matches:
            return matches[0]
    return None


def write_json(payload: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path


def read_json(input_path: Path) -> Any:
    return json.loads(input_path.read_text(encoding="utf-8"))


def write_dataframe(df: pd.DataFrame, output_path: Path, index: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)
    return output_path


def safe_numeric(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
