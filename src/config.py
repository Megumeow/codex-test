from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def _resolve_path(root_dir: Path, value: Any) -> Path | None:
    if value in (None, "", "null"):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (root_dir / path).resolve()
    return path


def _simple_yaml_load(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_list_key:
            result.setdefault(current_list_key, []).append(stripped[2:].strip())
            continue
        current_list_key = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            result[key] = []
            current_list_key = key
            continue
        lowered = value.lower()
        if lowered in {"true", "false"}:
            parsed: Any = lowered == "true"
        elif lowered in {"null", "none"}:
            parsed = None
        elif value == "[]":
            parsed = []
        else:
            try:
                parsed = int(value) if "." not in value else float(value)
            except ValueError:
                parsed = value
        result[key] = parsed
    return result


def _load_env_fallback(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(slots=True)
class AppConfig:
    root_dir: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    manual_gp_history_dir: Path
    outputs_dir: Path
    outputs_figures_dir: Path
    outputs_maps_dir: Path
    outputs_tables_dir: Path
    selected_norad_ids: list[int] = field(default_factory=list)
    case_selection_limit: int = 5
    selected_object_types: list[str] = field(default_factory=lambda: ["Payload", "Rocket Body", "Debris"])
    cords_page_url: str = "https://aerospace.org/reentries"
    cords_grid_url: str = "https://aerospace.org/reentries/grid"
    natural_earth_land_url: str = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
    natural_earth_countries_url: str = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    corridor_width_km: float = 200.0
    use_spacetrack_if_available: bool = True
    use_tle_track_if_available: bool = True
    use_population_raster_if_available: bool = False
    country_population_fallback: bool = True
    population_raster_path: Path | None = None
    country_population_csv: Path | None = None
    manual_path_file: Path | None = None
    track_duration_hours: float = 6.0
    track_step_minutes: int = 10
    min_training_rows: int = 24
    random_state: int = 42
    use_autogluon_if_available: bool = True
    autogluon_time_limit_seconds: int = 60
    autogluon_presets: str = "medium_quality"
    autogluon_model_candidates: str = "tree_ensemble"
    autogluon_enable_weighted_ensemble: bool = True

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.manual_gp_history_dir,
            self.outputs_dir,
            self.outputs_figures_dir,
            self.outputs_maps_dir,
            self.outputs_tables_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    root_dir = Path(__file__).resolve().parents[1]
    env_path = root_dir / ".env"
    if load_dotenv is not None:
        load_dotenv(env_path)
    else:
        _load_env_fallback(env_path)

    config_path = Path(config_path) if config_path else root_dir / "configs" / "default.yaml"
    config_path = config_path.resolve()
    config_text = config_path.read_text(encoding="utf-8")
    data = (yaml.safe_load(config_text) if yaml is not None else _simple_yaml_load(config_text)) or {}

    outputs_dir = _resolve_path(root_dir, data.get("outputs_dir", "outputs")) or (root_dir / "outputs")
    cfg = AppConfig(
        root_dir=root_dir,
        data_dir=root_dir / "data",
        raw_dir=root_dir / "data" / "raw",
        processed_dir=root_dir / "data" / "processed",
        manual_gp_history_dir=root_dir / "data" / "manual_gp_history",
        outputs_dir=outputs_dir,
        outputs_figures_dir=outputs_dir / "figures",
        outputs_maps_dir=outputs_dir / "maps",
        outputs_tables_dir=outputs_dir / "tables",
        selected_norad_ids=[int(item) for item in data.get("selected_norad_ids", [])],
        case_selection_limit=int(data.get("case_selection_limit", 5)),
        selected_object_types=list(data.get("selected_object_types", ["Payload", "Rocket Body", "Debris"])),
        cords_page_url=str(data.get("cords_page_url", "https://aerospace.org/reentries")),
        cords_grid_url=str(data.get("cords_grid_url", "https://aerospace.org/reentries/grid")),
        natural_earth_land_url=str(data.get("natural_earth_land_url", "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip")),
        natural_earth_countries_url=str(
            data.get("natural_earth_countries_url", "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip")
        ),
        corridor_width_km=float(data.get("corridor_width_km", 200.0)),
        use_spacetrack_if_available=bool(data.get("use_spacetrack_if_available", True)),
        use_tle_track_if_available=bool(data.get("use_tle_track_if_available", True)),
        use_population_raster_if_available=bool(data.get("use_population_raster_if_available", False)),
        country_population_fallback=bool(data.get("country_population_fallback", True)),
        population_raster_path=_resolve_path(root_dir, data.get("population_raster_path")),
        country_population_csv=_resolve_path(root_dir, data.get("country_population_csv")),
        manual_path_file=_resolve_path(root_dir, data.get("manual_path_file", "data/sample_path.csv")),
        track_duration_hours=float(data.get("track_duration_hours", 6.0)),
        track_step_minutes=int(data.get("track_step_minutes", 10)),
        min_training_rows=int(data.get("min_training_rows", 24)),
        random_state=int(data.get("random_state", 42)),
        use_autogluon_if_available=bool(data.get("use_autogluon_if_available", True)),
        autogluon_time_limit_seconds=int(data.get("autogluon_time_limit_seconds", 60)),
        autogluon_presets=str(data.get("autogluon_presets", "medium_quality")),
        autogluon_model_candidates=str(data.get("autogluon_model_candidates", "tree_ensemble")),
        autogluon_enable_weighted_ensemble=bool(data.get("autogluon_enable_weighted_ensemble", True)),
    )
    cfg.ensure_directories()
    return cfg
