from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd

from src.config import AppConfig
from src.io_utils import download_file, fetch_text, requests_session, write_dataframe


LOGGER = logging.getLogger(__name__)


def _clean_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _match_column(columns: Iterable[str], *keywords: str) -> str | None:
    normalized = {column: _clean_column_name(column) for column in columns}
    for column, cleaned in normalized.items():
        if all(keyword in cleaned for keyword in keywords):
            return column
    return None


def _extract_history_csv_url(page_html: str, base_url: str) -> str | None:
    patterns = [
        r"https?://[^\s\"']*Reentry_History_Spreadsheet_[^\"']+\.csv",
        r"/sites/default/files/[^\s\"']*Reentry_History_Spreadsheet_[^\"']+\.csv",
    ]
    for pattern in patterns:
        match = re.search(pattern, page_html, flags=re.IGNORECASE)
        if match:
            return urljoin(base_url, match.group(0))
    return None


def _guess_history_csv_url(lookback_days: int = 30) -> str | None:
    session = requests_session()
    today = datetime.now(timezone.utc).date()
    for offset in range(0, lookback_days + 1):
        candidate_date = today - timedelta(days=offset)
        url = (
            "https://aerospace.org/sites/default/files/"
            f"{candidate_date:%Y-%m}/Reentry_History_Spreadsheet_{candidate_date:%m-%d-%y}.csv"
        )
        try:
            response = session.get(url, timeout=30, stream=True)
            if response.status_code == 200:
                response.close()
                return url
            response.close()
        except Exception:
            continue
    return None


def _parse_name_and_norad(value: object) -> tuple[str | None, int | None]:
    if pd.isna(value):
        return None, None
    text = str(value).strip()
    match = re.search(r"(?:NORAD|CAT(?:ALOG)?|ID)?\s*[:#]?\s*(\d{3,6})", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"[\[(](\d{3,6})[\])]", text)
    norad_id = int(match.group(1)) if match else None
    cleaned = re.sub(r"[\[(].*?\d{3,6}.*?[\])]", "", text).strip(" -|")
    return cleaned or text, norad_id


def _parse_datetime_column(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"\s*±\s*[0-9.]+\s*hours?.*$", "", regex=True)
        .str.replace(r"\s+UTC$", "", regex=True)
        .str.strip()
        .replace({"nan": None, "NaT": None, "Unknown": None, "TBD": None})
    )
    return pd.to_datetime(cleaned, utc=True, errors="coerce")


def _parse_uncertainty_hours(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"±\s*([0-9.]+)\s*hours?", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _normalize_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["object_name", "norad_id", "reentry_time_utc", "object_type", "source", "launch_date"])

    object_col = _match_column(df.columns, "object") or _match_column(df.columns, "mission") or _match_column(df.columns, "name")
    norad_col = _match_column(df.columns, "norad") or _match_column(df.columns, "catalog") or _match_column(df.columns, "cat", "id") or _match_column(df.columns, "ssn")
    reentry_col = (
        _match_column(df.columns, "predicted", "reentry")
        or _match_column(df.columns, "reentry")
        or _match_column(df.columns, "decay")
    )
    launch_col = _match_column(df.columns, "launch", "date") or _match_column(df.columns, "launched") or _match_column(df.columns, "launch")
    object_type_col = _match_column(df.columns, "object", "type") or _match_column(df.columns, "reentry", "type") or _match_column(df.columns, "type")

    normalized = pd.DataFrame(index=df.index)
    if object_col:
        parsed = df[object_col].apply(_parse_name_and_norad)
        normalized["object_name"] = parsed.apply(lambda item: item[0])
        normalized["norad_from_name"] = parsed.apply(lambda item: item[1])
    else:
        normalized["object_name"] = None
        normalized["norad_from_name"] = None

    if norad_col:
        normalized["norad_id"] = pd.to_numeric(df[norad_col], errors="coerce").fillna(normalized["norad_from_name"]).astype("Int64")
    else:
        normalized["norad_id"] = pd.Series(normalized["norad_from_name"], dtype="Int64")

    if reentry_col:
        normalized["reentry_time_utc"] = _parse_datetime_column(df[reentry_col])
        normalized["reentry_uncertainty_hours"] = _parse_uncertainty_hours(df[reentry_col])
    else:
        date_col = _match_column(df.columns, "reentry", "date") or _match_column(df.columns, "decay", "date")
        time_col = _match_column(df.columns, "reentry", "time") or _match_column(df.columns, "decay", "time")
        if date_col and time_col:
            combined = df[date_col].astype(str) + " " + df[time_col].astype(str)
            normalized["reentry_time_utc"] = _parse_datetime_column(combined)
            normalized["reentry_uncertainty_hours"] = pd.NA
        else:
            normalized["reentry_time_utc"] = pd.NaT
            normalized["reentry_uncertainty_hours"] = pd.NA

    normalized["object_type"] = df[object_type_col].astype(str).str.strip() if object_type_col else None
    normalized["launch_date"] = pd.to_datetime(df[launch_col], utc=True, errors="coerce") if launch_col else pd.NaT
    normalized["source"] = source
    normalized["source_object_raw"] = df[object_col].astype(str).str.strip() if object_col else None
    normalized = normalized.drop(columns=["norad_from_name"], errors="ignore")
    return normalized


def download_cords_sources(config: AppConfig, force: bool = False) -> dict[str, Path]:
    cords_dir = config.raw_dir / "cords"
    cords_dir.mkdir(parents=True, exist_ok=True)

    page_path = cords_dir / "reentries_page.html"
    grid_path = cords_dir / "reentries_grid.html"
    history_path = cords_dir / "reentry_history.csv"

    page_html = ""
    try:
        page_html = fetch_text(config.cords_page_url, target_path=page_path, force=force)
    except Exception as exc:
        LOGGER.warning("Could not fetch CORDS page directly: %s", exc)

    try:
        fetch_text(config.cords_grid_url, target_path=grid_path, force=force)
    except Exception as exc:
        LOGGER.warning("Could not fetch CORDS grid page directly: %s", exc)

    history_url = _extract_history_csv_url(page_html, config.cords_page_url) if page_html else None
    if history_url is None:
        history_url = _guess_history_csv_url()
    if history_url:
        download_file(history_url, history_path, force=force)
    else:
        LOGGER.warning("Could not find historical CORDS CSV link on %s", config.cords_page_url)

    return {"page": page_path, "grid": grid_path, "history_csv": history_path}


def load_cords_reentries(config: AppConfig, force: bool = False) -> pd.DataFrame:
    sources = download_cords_sources(config, force=force)
    frames: list[pd.DataFrame] = []

    history_path = sources["history_csv"]
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        frames.append(_normalize_frame(history_df, source="aerospace_cords_history_csv"))

    grid_path = sources["grid"]
    if grid_path.exists():
        try:
            grid_tables = pd.read_html(grid_path)
            for idx, table in enumerate(grid_tables):
                normalized = _normalize_frame(table, source=f"aerospace_cords_grid_{idx}")
                if normalized["reentry_time_utc"].notna().any() or normalized["object_name"].notna().any():
                    frames.append(normalized)
                    break
        except Exception as exc:
            LOGGER.warning("Could not parse CORDS grid page: %s", exc)

    if not frames:
        raise RuntimeError("No CORDS reentry tables could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined["object_name"] = combined["object_name"].fillna(combined.get("source_object_raw")).astype(str).str.strip()
    combined["object_type"] = combined["object_type"].replace({"nan": None, "None": None})
    combined = combined.drop_duplicates(subset=["object_name", "norad_id", "reentry_time_utc"], keep="first")
    combined = combined.sort_values(["reentry_time_utc", "object_name"], ascending=[False, True], na_position="last").reset_index(drop=True)

    processed_path = config.processed_dir / "reentries_clean.csv"
    output_path = config.outputs_tables_dir / "reentries_clean.csv"
    write_dataframe(combined, processed_path)
    write_dataframe(combined, output_path)
    return combined


def select_presentation_cases(reentries: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    df = reentries.copy()
    df = df[df["norad_id"].notna() & df["reentry_time_utc"].notna()].copy()
    df["norad_id"] = df["norad_id"].astype(int)

    if config.selected_object_types:
        allowed = {item.lower() for item in config.selected_object_types}
        mask = df["object_type"].fillna("").str.lower().isin(allowed) | df["object_type"].isna()
        df = df[mask].copy()

    if config.selected_norad_ids:
        selected = df[df["norad_id"].isin(config.selected_norad_ids)].copy()
    else:
        selected_parts: list[pd.DataFrame] = []
        for object_type in config.selected_object_types:
            subset = df[df["object_type"].fillna("").str.lower() == object_type.lower()]
            if not subset.empty:
                selected_parts.append(subset.head(1))
        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=df.columns)
        remaining = df[~df["norad_id"].isin(selected["norad_id"].tolist())]
        fill_count = max(config.case_selection_limit - len(selected), 0)
        if fill_count:
            selected = pd.concat([selected, remaining.head(fill_count)], ignore_index=True)

    selected = selected.drop_duplicates(subset=["norad_id"]).head(config.case_selection_limit).copy()
    selected.insert(0, "case_id", [f"case_{idx + 1:02d}" for idx in range(len(selected))])
    output_path = config.outputs_tables_dir / "selected_cases.csv"
    write_dataframe(selected, output_path)
    return selected
