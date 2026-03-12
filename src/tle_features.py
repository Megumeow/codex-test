from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
MU_EARTH_KM3_S2 = 398600.4418


def _clean_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {_clean_column_name(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for cleaned, original in normalized.items():
        if any(candidate in cleaned for candidate in candidates):
            return original
    return None


def tle_exponent_to_float(value: str) -> float | None:
    text = value.strip()
    if not text or text == "00000-0":
        return 0.0
    sign = "-" if text.startswith("-") else ""
    mantissa = text[1:6] if text[0] in "+-" else text[:5]
    exponent = text[-2:]
    try:
        return float(f"{sign}0.{mantissa}e{int(exponent):+d}")
    except ValueError:
        return None


def tle_epoch_to_datetime(epoch_year: str, epoch_day: str) -> datetime:
    year = int(epoch_year)
    year += 2000 if year < 57 else 1900
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    return start + timedelta(days=float(epoch_day) - 1.0)


def parse_tle_history_text(text: str, source_file: Path) -> pd.DataFrame:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    records: list[dict[str, object]] = []
    pending_name: str | None = None
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("1 ") and idx + 1 < len(lines) and lines[idx + 1].startswith("2 "):
            line1 = line
            line2 = lines[idx + 1]
            norad = int(line1[2:7])
            epoch = tle_epoch_to_datetime(line1[18:20], line1[20:32])
            records.append(
                {
                    "norad_id": norad,
                    "object_name": pending_name,
                    "epoch": epoch,
                    "mean_motion": float(line2[52:63].strip()),
                    "bstar": tle_exponent_to_float(line1[53:61]),
                    "eccentricity": float(f"0.{line2[26:33].strip()}"),
                    "inclination": float(line2[8:16].strip()),
                    "raan": float(line2[17:25].strip()),
                    "arg_perigee": float(line2[34:42].strip()),
                    "mean_anomaly": float(line2[43:51].strip()),
                    "tle_line1": line1,
                    "tle_line2": line2,
                    "source_file": str(source_file),
                }
            )
            pending_name = None
            idx += 2
            continue
        pending_name = line.strip()
        idx += 1
    return pd.DataFrame(records)


def normalize_gp_history_frame(df: pd.DataFrame, source_file: str | Path | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "norad_id",
                "object_name",
                "epoch",
                "mean_motion",
                "bstar",
                "eccentricity",
                "inclination",
                "raan",
                "arg_perigee",
                "mean_anomaly",
                "tle_line1",
                "tle_line2",
                "source_file",
            ]
        )

    epoch_col = _find_column(df.columns, ["epoch", "epoch_datetime", "tle_epoch"])
    mean_motion_col = _find_column(df.columns, ["mean_motion", "meanmotion", "no_kozai"])
    bstar_col = _find_column(df.columns, ["bstar", "b_star"])
    eccentricity_col = _find_column(df.columns, ["eccentricity", "ecc"])
    inclination_col = _find_column(df.columns, ["inclination", "inclo"])
    raan_col = _find_column(df.columns, ["raan", "nodeo"])
    arg_perigee_col = _find_column(df.columns, ["arg_perigee", "argpo"])
    mean_anomaly_col = _find_column(df.columns, ["mean_anomaly", "mo"])
    norad_col = _find_column(df.columns, ["norad_cat_id", "norad_id", "satnum", "catalog_number"])
    object_name_col = _find_column(df.columns, ["object_name", "satname", "name"])
    tle_line1_col = _find_column(df.columns, ["tle_line1", "line1"])
    tle_line2_col = _find_column(df.columns, ["tle_line2", "line2"])

    normalized = pd.DataFrame(index=df.index)
    normalized["norad_id"] = pd.to_numeric(df[norad_col], errors="coerce").astype("Int64") if norad_col else pd.Series(dtype="Int64")
    normalized["object_name"] = df[object_name_col].astype(str).str.strip() if object_name_col else None
    normalized["epoch"] = pd.to_datetime(df[epoch_col], utc=True, errors="coerce") if epoch_col else pd.NaT
    normalized["mean_motion"] = pd.to_numeric(df[mean_motion_col], errors="coerce") if mean_motion_col else np.nan
    normalized["bstar"] = pd.to_numeric(df[bstar_col], errors="coerce") if bstar_col else np.nan
    normalized["eccentricity"] = pd.to_numeric(df[eccentricity_col], errors="coerce") if eccentricity_col else np.nan
    normalized["inclination"] = pd.to_numeric(df[inclination_col], errors="coerce") if inclination_col else np.nan
    normalized["raan"] = pd.to_numeric(df[raan_col], errors="coerce") if raan_col else np.nan
    normalized["arg_perigee"] = pd.to_numeric(df[arg_perigee_col], errors="coerce") if arg_perigee_col else np.nan
    normalized["mean_anomaly"] = pd.to_numeric(df[mean_anomaly_col], errors="coerce") if mean_anomaly_col else np.nan
    normalized["tle_line1"] = df[tle_line1_col].astype(str).where(df[tle_line1_col].notna(), None) if tle_line1_col else None
    normalized["tle_line2"] = df[tle_line2_col].astype(str).where(df[tle_line2_col].notna(), None) if tle_line2_col else None
    normalized["source_file"] = str(source_file) if source_file is not None else None
    normalized = normalized.dropna(subset=["epoch", "mean_motion"], how="any").copy()
    normalized = normalized.sort_values("epoch").reset_index(drop=True)
    return normalized


def load_manual_gp_history(manual_dir: Path, norad_ids: list[int] | None = None) -> dict[int, pd.DataFrame]:
    files = sorted([path for path in manual_dir.iterdir() if path.is_file() and path.suffix.lower() in {".csv", ".json", ".txt", ".tle"}])
    outputs: dict[int, pd.DataFrame] = {}
    for file_path in files:
        if file_path.suffix.lower() == ".csv":
            frame = normalize_gp_history_frame(pd.read_csv(file_path), source_file=file_path)
        elif file_path.suffix.lower() == ".json":
            frame = normalize_gp_history_frame(pd.read_json(file_path), source_file=file_path)
        else:
            frame = parse_tle_history_text(file_path.read_text(encoding="utf-8"), file_path)
            frame = normalize_gp_history_frame(frame, source_file=file_path)
        if frame.empty:
            continue
        if frame["norad_id"].isna().all():
            match = re.search(r"(\d{3,6})", file_path.stem)
            if match:
                frame["norad_id"] = int(match.group(1))
        frame = frame.dropna(subset=["norad_id"]).copy()
        frame["norad_id"] = frame["norad_id"].astype(int)
        for norad_id, subset in frame.groupby("norad_id"):
            if norad_ids and norad_id not in norad_ids:
                continue
            outputs[norad_id] = pd.concat([outputs.get(norad_id, pd.DataFrame()), subset], ignore_index=True)
    return {norad_id: df.sort_values("epoch").drop_duplicates(subset=["epoch"]).reset_index(drop=True) for norad_id, df in outputs.items()}


def semi_major_axis_from_mean_motion(mean_motion_rev_per_day: pd.Series) -> pd.Series:
    mean_motion_rad_s = mean_motion_rev_per_day * (2.0 * math.pi) / 86400.0
    return (MU_EARTH_KM3_S2 / np.power(mean_motion_rad_s, 2.0)) ** (1.0 / 3.0)


def _rolling_slope(values: pd.Series, times: pd.Series, window: int = 3) -> pd.Series:
    slopes = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        y = values.iloc[start : idx + 1]
        x = times.iloc[start : idx + 1]
        if len(y.dropna()) < 2 or len(x.dropna()) < 2:
            slopes.append(np.nan)
            continue
        x_hours = (x - x.iloc[0]).dt.total_seconds() / 3600.0
        if np.isclose(x_hours.iloc[-1], 0.0):
            slopes.append(np.nan)
            continue
        slope = np.polyfit(x_hours, y, 1)[0]
        slopes.append(slope)
    return pd.Series(slopes, index=values.index)


def build_feature_table(gp_history_df: pd.DataFrame, reentry_time_utc: pd.Timestamp) -> pd.DataFrame:
    df = gp_history_df.copy().sort_values("epoch").reset_index(drop=True)
    df["delta_hours"] = df["epoch"].diff().dt.total_seconds() / 3600.0
    df["delta_mean_motion"] = df["mean_motion"].diff()
    df["delta_bstar"] = df["bstar"].diff()
    df["semi_major_axis_km"] = semi_major_axis_from_mean_motion(df["mean_motion"])
    df["hours_to_decay"] = (pd.Timestamp(reentry_time_utc) - df["epoch"]).dt.total_seconds() / 3600.0
    df["mean_motion_slope_per_hour"] = _rolling_slope(df["mean_motion"], df["epoch"], window=3)
    df["bstar_slope_per_hour"] = _rolling_slope(df["bstar"].fillna(0.0), df["epoch"], window=3)
    df["semi_major_axis_slope_km_per_hour"] = _rolling_slope(df["semi_major_axis_km"], df["epoch"], window=3)
    df["epoch_dayofyear"] = df["epoch"].dt.dayofyear
    df["epoch_hour"] = df["epoch"].dt.hour + df["epoch"].dt.minute / 60.0
    df = df[df["hours_to_decay"] >= 0].copy()
    return df
