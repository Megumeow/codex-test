from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.cords_loader import load_cords_reentries, select_presentation_cases
from src.corridor import build_corridor_from_points, build_path_from_tle_history, load_path_points, save_corridor_geojson
from src.exposure import run_exposure_analysis
from src.io_utils import configure_logging
from src.plotting import (
    plot_corridor_static,
    plot_country_overlap,
    plot_land_ocean,
    plot_population_summary,
    plot_time_window_diagnostics,
    save_corridor_map,
)
from src.spacetrack_client import collect_gp_history
from src.time_window_model import run_time_window_model


def _load_or_build_reentries(config, force_download: bool) -> pd.DataFrame:
    cached = config.outputs_tables_dir / "reentries_clean.csv"
    if cached.exists() and not force_download:
        return pd.read_csv(cached, parse_dates=["reentry_time_utc", "launch_date"])
    return load_cords_reentries(config, force=force_download)


def _select_cases(config, reentries: pd.DataFrame) -> pd.DataFrame:
    selected_path = config.outputs_tables_dir / "selected_cases.csv"
    if selected_path.exists():
        return pd.read_csv(selected_path, parse_dates=["reentry_time_utc", "launch_date"])
    return select_presentation_cases(reentries, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reduced offline reentry exposure demo pipeline.")
    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    parser.add_argument("--path-file", default=None, help="Optional CSV or GeoJSON path input.")
    parser.add_argument("--case-norad", type=int, default=None, help="Optional NORAD ID to prioritize.")
    parser.add_argument("--width-km", type=float, default=None, help="Override corridor width in kilometers.")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of public datasets.")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    width_km = float(args.width_km) if args.width_km is not None else config.corridor_width_km

    reentries = _load_or_build_reentries(config, force_download=args.force_download)
    selected_cases = _select_cases(config, reentries)
    if args.case_norad is not None:
        filtered = selected_cases[selected_cases["norad_id"].astype("Int64") == int(args.case_norad)]
        if not filtered.empty:
            selected_cases = filtered.reset_index(drop=True)
    if selected_cases.empty:
        raise RuntimeError("No presentation cases are available after filtering.")

    gp_histories = collect_gp_history(selected_cases, config, force=False)
    model_result = run_time_window_model(gp_histories, selected_cases, config)
    if model_result is not None:
        plot_time_window_diagnostics(
            predictions=model_result.predictions,
            metrics=model_result.metrics,
            feature_importance=model_result.feature_importance,
            output_dir=config.outputs_figures_dir,
        )

    path_points = None
    selected_case = selected_cases.iloc[0]
    case_norad = int(selected_case["norad_id"])
    if config.use_tle_track_if_available and case_norad in gp_histories:
        reentry_time = pd.to_datetime(selected_case["reentry_time_utc"], utc=True, errors="coerce")
        path_points = build_path_from_tle_history(
            gp_histories[case_norad],
            reentry_time_utc=reentry_time,
            track_duration_hours=config.track_duration_hours,
            track_step_minutes=config.track_step_minutes,
        )

    if path_points is None:
        input_path = Path(args.path_file).resolve() if args.path_file else config.manual_path_file
        if input_path is None or not input_path.exists():
            raise FileNotFoundError("No path file was available for corridor generation.")
        path_points = load_path_points(input_path)

    corridor_gdf, path_points_gdf = build_corridor_from_points(path_points, width_km=width_km)
    save_corridor_geojson(corridor_gdf, config.outputs_maps_dir / "corridor.geojson")
    summary_df, country_overlap, land, countries = run_exposure_analysis(corridor_gdf, config)
    map_title = f"Reentry Corridor: {selected_case['object_name']} ({case_norad})"
    plot_corridor_static(
        corridor_gdf,
        config.outputs_figures_dir / "corridor_static.png",
        land,
        countries,
        path_points_gdf,
        summary_df=summary_df,
        country_overlap=country_overlap,
        title=map_title,
    )
    save_corridor_map(
        corridor_gdf,
        config.outputs_maps_dir / "corridor.html",
        path_points_gdf,
        summary_df=summary_df,
        country_overlap=country_overlap,
        map_title=map_title,
    )
    plot_country_overlap(country_overlap, config.outputs_figures_dir / "country_overlap_top10.png")
    plot_land_ocean(summary_df, config.outputs_figures_dir / "land_ocean_fraction.png")
    plot_population_summary(summary_df, config.outputs_figures_dir / "population_exposure_summary.png")

    print("Reduced reentry exposure demo complete.")
    print(f"Selected case: {selected_case['object_name']} ({case_norad})")
    print(f"Corridor area km^2: {summary_df.iloc[0]['corridor_area_km2']:.2f}")
    if pd.notna(summary_df.iloc[0]["land_fraction"]):
        print(f"Land fraction: {summary_df.iloc[0]['land_fraction']:.3f}")
        print(f"Ocean fraction: {summary_df.iloc[0]['ocean_fraction']:.3f}")
    print(f"Exposure summary: {config.outputs_tables_dir / 'exposure_summary.csv'}")
    print(f"Corridor map: {config.outputs_maps_dir / 'corridor.html'}")


if __name__ == "__main__":
    main()
