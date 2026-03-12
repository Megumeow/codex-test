from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import AppConfig
from src.io_utils import requests_session, write_dataframe
from src.tle_features import load_manual_gp_history, normalize_gp_history_frame


LOGGER = logging.getLogger(__name__)


SPACE_TRACK_LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
SPACE_TRACK_GP_HISTORY_URL = (
    "https://www.space-track.org/basicspacedata/query/class/gp_history/"
    "NORAD_CAT_ID/{norad_id}/orderby/EPOCH asc/format/json"
)


@dataclass(slots=True)
class SpaceTrackClient:
    username: str
    password: str

    @classmethod
    def from_env(cls) -> SpaceTrackClient | None:
        username = os.getenv("SPACETRACK_USER")
        password = os.getenv("SPACETRACK_PASS")
        if not username or not password:
            return None
        return cls(username=username, password=password)

    def _login_session(self):
        session = requests_session()
        response = session.post(
            SPACE_TRACK_LOGIN_URL,
            data={"identity": self.username, "password": self.password},
            timeout=60,
        )
        response.raise_for_status()
        return session

    def download_gp_history(self, norad_id: int, target_path: Path, force: bool = False) -> Path:
        if target_path.exists() and not force:
            LOGGER.info("Using cached GP history for NORAD %s", norad_id)
            return target_path

        session = self._login_session()
        url = SPACE_TRACK_GP_HISTORY_URL.format(norad_id=int(norad_id))
        LOGGER.info("Downloading Space-Track GP history for NORAD %s", norad_id)
        response = session.get(url, timeout=120)
        response.raise_for_status()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(response.text, encoding="utf-8")
        return target_path


def _load_spacetrack_json(file_path: Path) -> pd.DataFrame:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected Space-Track response structure in {file_path}")
    return pd.DataFrame(payload)


def collect_gp_history(
    selected_cases: pd.DataFrame,
    config: AppConfig,
    force: bool = False,
) -> dict[int, pd.DataFrame]:
    norad_ids = sorted({int(value) for value in selected_cases["norad_id"].dropna().astype(int).tolist()})
    outputs: dict[int, pd.DataFrame] = {}

    client = SpaceTrackClient.from_env() if config.use_spacetrack_if_available else None
    if client is None:
        LOGGER.info("Space-Track credentials not available. Falling back to local manual GP history if present.")

    raw_dir = config.raw_dir / "spacetrack"
    processed_dir = config.processed_dir / "gp_history"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if client is not None:
        for norad_id in norad_ids:
            raw_file = raw_dir / f"norad_{norad_id}_gp_history.json"
            try:
                client.download_gp_history(norad_id=norad_id, target_path=raw_file, force=force)
                frame = normalize_gp_history_frame(_load_spacetrack_json(raw_file), source_file=raw_file)
                if not frame.empty:
                    frame["norad_id"] = frame["norad_id"].fillna(norad_id).astype(int)
                    outputs[norad_id] = frame.sort_values("epoch").reset_index(drop=True)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to download GP history for NORAD %s: %s", norad_id, exc)

    manual_data = load_manual_gp_history(config.manual_gp_history_dir, norad_ids=norad_ids)
    for norad_id, frame in manual_data.items():
        outputs[norad_id] = (
            pd.concat([outputs.get(norad_id, pd.DataFrame()), frame], ignore_index=True)
            .sort_values("epoch")
            .drop_duplicates(subset=["epoch"])
            .reset_index(drop=True)
        )

    for norad_id, frame in outputs.items():
        processed_path = processed_dir / f"norad_{norad_id}_gp_history.csv"
        write_dataframe(frame, processed_path)

    return outputs
