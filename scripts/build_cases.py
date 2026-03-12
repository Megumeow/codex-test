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
from src.io_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a small set of presentation cases from the CORDS metadata.")
    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    parser.add_argument("--force", action="store_true", help="Force refresh of the cleaned reentry table.")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)

    reentries_path = config.outputs_tables_dir / "reentries_clean.csv"
    if reentries_path.exists() and not args.force:
        reentries = pd.read_csv(reentries_path, parse_dates=["reentry_time_utc", "launch_date"])
    else:
        reentries = load_cords_reentries(config, force=args.force)
    selected = select_presentation_cases(reentries, config)

    print(f"Selected {len(selected)} cases.")
    print(selected[["case_id", "object_name", "norad_id", "reentry_time_utc"]].to_string(index=False))


if __name__ == "__main__":
    main()
