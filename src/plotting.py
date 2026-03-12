from __future__ import annotations

from pathlib import Path

import branca
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


def _format_large_number(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "n/a"
    numeric = float(numeric)
    if abs(numeric) >= 1_000_000_000:
        return f"{numeric / 1_000_000_000:.2f}B"
    if abs(numeric) >= 1_000_000:
        return f"{numeric / 1_000_000:.2f}M"
    if abs(numeric) >= 1_000:
        return f"{numeric / 1_000:.1f}K"
    return f"{numeric:.0f}"


def _summary_lines(summary_df: pd.DataFrame) -> list[str]:
    if summary_df.empty:
        return ["No exposure summary available"]
    row = summary_df.iloc[0]
    lines = [
        f"Corridor area: {_format_large_number(row.get('corridor_area_km2'))} km^2",
        f"Land fraction: {float(row.get('land_fraction', 0.0)):.1%}" if pd.notna(row.get("land_fraction")) else "Land fraction: n/a",
        f"Ocean fraction: {float(row.get('ocean_fraction', 0.0)):.1%}" if pd.notna(row.get("ocean_fraction")) else "Ocean fraction: n/a",
    ]
    if pd.notna(row.get("raster_population_exposure")):
        lines.append(f"Raster population: {_format_large_number(row.get('raster_population_exposure'))}")
    elif pd.notna(row.get("coarse_population_exposure_score_total")):
        lines.append(f"Coarse population score: {_format_large_number(row.get('coarse_population_exposure_score_total'))}")
    method = row.get("population_method")
    if pd.notna(method):
        lines.append(f"Population method: {method}")
    return lines


def _top_country_lines(country_overlap: pd.DataFrame, top_n: int = 3) -> list[str]:
    if country_overlap.empty:
        return ["Top countries: n/a"]
    lines = ["Top countries:"]
    for _, row in country_overlap.head(top_n).iterrows():
        lines.append(f"{row['country_name']}: {row['corridor_overlap_fraction']:.1%}")
    return lines


def plot_corridor_static(
    corridor_gdf: gpd.GeoDataFrame,
    output_path: Path,
    land_gdf: gpd.GeoDataFrame | None = None,
    countries_gdf: gpd.GeoDataFrame | None = None,
    path_points_gdf: gpd.GeoDataFrame | None = None,
    summary_df: pd.DataFrame | None = None,
    country_overlap: pd.DataFrame | None = None,
    title: str = "Simplified Reentry Corridor",
) -> Path:
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_facecolor("#f4f6f8")
    if land_gdf is not None and not land_gdf.empty:
        land_gdf.to_crs("+proj=eqearth").plot(ax=ax, color="#d9e2d0", edgecolor="none")
    if countries_gdf is not None and not countries_gdf.empty:
        countries_gdf.to_crs("+proj=eqearth").boundary.plot(ax=ax, color="#9aa6b2", linewidth=0.3)
    corridor_gdf.to_crs("+proj=eqearth").plot(ax=ax, color="#d1495b", alpha=0.35, edgecolor="#8f1024", linewidth=1.5)
    if path_points_gdf is not None and not path_points_gdf.empty:
        path_points_gdf.to_crs("+proj=eqearth").plot(ax=ax, color="#1f4e79", markersize=10)

    if summary_df is not None:
        panel_lines = _summary_lines(summary_df)
        if country_overlap is not None:
            panel_lines.extend(_top_country_lines(country_overlap))
        ax.text(
            0.02,
            0.98,
            "\n".join(panel_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="#16212e",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#b8c4d0", "alpha": 0.92},
        )

    ax.set_title(title, fontsize=18)
    ax.set_axis_off()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_corridor_map(
    corridor_gdf: gpd.GeoDataFrame,
    output_path: Path,
    path_points_gdf: gpd.GeoDataFrame | None = None,
    summary_df: pd.DataFrame | None = None,
    country_overlap: pd.DataFrame | None = None,
    map_title: str | None = None,
) -> Path:
    centroid = corridor_gdf.to_crs("EPSG:4326").geometry.iloc[0].centroid
    fmap = folium.Map(location=[centroid.y, centroid.x], zoom_start=2, tiles="CartoDB positron")

    if map_title:
        title_html = f"""
        <div style="position: fixed; top: 14px; left: 50px; z-index: 9999;
                    background: rgba(255,255,255,0.92); padding: 10px 14px;
                    border: 1px solid #b8c4d0; border-radius: 8px;
                    font-family: Arial, sans-serif; font-size: 16px; color: #16212e;">
            <b>{map_title}</b>
        </div>
        """
        fmap.get_root().html.add_child(branca.element.Element(title_html))

    folium.GeoJson(
        corridor_gdf.to_json(),
        name="Corridor",
        style_function=lambda _: {"fillColor": "#d1495b", "color": "#8f1024", "weight": 2, "fillOpacity": 0.3},
    ).add_to(fmap)

    if path_points_gdf is not None and not path_points_gdf.empty:
        coordinates = [[geom.y, geom.x] for geom in path_points_gdf.geometry]
        folium.PolyLine(coordinates, color="#1f4e79", weight=3, opacity=0.9).add_to(fmap)

    if summary_df is not None and not summary_df.empty:
        summary_html = "<br>".join(_summary_lines(summary_df))
        if country_overlap is not None:
            summary_html += "<br><br>" + "<br>".join(_top_country_lines(country_overlap))
        panel_html = f"""
        <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                    width: 310px; background: rgba(255,255,255,0.94);
                    border: 1px solid #b8c4d0; border-radius: 8px; padding: 12px;
                    font-family: Arial, sans-serif; font-size: 12px; color: #16212e;">
            <b>Exposure Summary</b><br>{summary_html}
        </div>
        """
        fmap.get_root().html.add_child(branca.element.Element(panel_html))

    folium.LayerControl().add_to(fmap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_path))
    return output_path


def plot_country_overlap(country_overlap: pd.DataFrame, output_path: Path, top_n: int = 10) -> Path | None:
    if country_overlap.empty:
        return None
    top = country_overlap.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["country_name"], top["overlap_area_km2"], color="#1f4e79")
    ax.invert_yaxis()
    ax.set_xlabel("Overlap Area (km^2)")
    ax.set_ylabel("")
    ax.set_title("Top Countries By Corridor Overlap")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_land_ocean(summary_df: pd.DataFrame, output_path: Path) -> Path | None:
    if summary_df.empty or pd.isna(summary_df.iloc[0].get("land_fraction")):
        return None
    land_fraction = float(summary_df.iloc[0]["land_fraction"])
    ocean_fraction = float(summary_df.iloc[0]["ocean_fraction"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        [land_fraction, ocean_fraction],
        labels=["Land", "Ocean"],
        colors=["#8fb996", "#5b8fb9"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Land vs Ocean Fraction")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_population_summary(summary_df: pd.DataFrame, output_path: Path) -> Path | None:
    if summary_df.empty:
        return None
    row = summary_df.iloc[0]
    coarse = pd.to_numeric(pd.Series([row.get("coarse_population_exposure_score_total")]), errors="coerce").iloc[0]
    raster = pd.to_numeric(pd.Series([row.get("raster_population_exposure")]), errors="coerce").iloc[0]
    if pd.isna(coarse) and pd.isna(raster):
        return None

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    if pd.notna(coarse):
        labels.append("Country fallback")
        values.append(float(coarse))
        colors.append("#5b8fb9")
    if pd.notna(raster):
        labels.append("Raster zonal sum")
        values.append(float(raster))
        colors.append("#d1495b")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_ylabel("Population exposure")
    ax.set_title("Population Exposure Summary")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, _format_large_number(value), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_time_window_diagnostics(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    saved: list[Path] = []
    if predictions.empty:
        return saved

    compare = predictions[predictions["split"].isin(["test", "latest_rows"])].copy()
    if not compare.empty:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(compare["actual_hours_to_decay"], compare["predicted_hours_to_decay"], color="#d1495b", alpha=0.8)
        max_val = max(compare["actual_hours_to_decay"].max(), compare["predicted_hours_to_decay"].max())
        ax.plot([0, max_val], [0, max_val], linestyle="--", color="#444444")
        ax.set_xlabel("Actual Hours To Decay")
        ax.set_ylabel("Predicted Hours To Decay")
        ax.set_title("Predicted vs Actual Hours To Decay")
        fig.tight_layout()
        output_path = output_dir / "predicted_vs_actual_hours_to_decay.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(output_path)

        fig, ax = plt.subplots(figsize=(8, 5))
        errors = compare["predicted_hours_to_decay"] - compare["actual_hours_to_decay"]
        ax.hist(errors, bins=min(12, max(5, len(errors))), color="#5b8fb9", edgecolor="white")
        ax.set_xlabel("Prediction Error (hours)")
        ax.set_ylabel("Count")
        ax.set_title("Time-Window Prediction Error")
        fig.tight_layout()
        output_path = output_dir / "time_window_error_histogram.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(output_path)

    if not feature_importance.empty:
        top = feature_importance.head(10).sort_values("importance")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top["feature"], top["importance"], color="#8f1024")
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        ax.set_title("Random Forest Feature Importance")
        fig.tight_layout()
        output_path = output_dir / "time_window_feature_importance.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(output_path)

    if not metrics.empty:
        metrics_output = output_dir / "time_window_model_metrics.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        metrics.plot(x="model_name", y=["mae_hours", "rmse_hours"], kind="bar", ax=ax, color=["#1f4e79", "#d1495b"])
        ax.set_ylabel("Hours")
        ax.set_title("Time-Window Model Error")
        fig.tight_layout()
        fig.savefig(metrics_output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(metrics_output)
    return saved
