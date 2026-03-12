from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.cords_loader import load_cords_reentries
from src.exposure import ensure_natural_earth_layers
from src.io_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and cache public reentry prototype datasets.")
    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    parser.add_argument("--force", action="store_true", help="Force re-download of cached files.")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    reentries = load_cords_reentries(config, force=args.force)
    ne_paths = ensure_natural_earth_layers(config, force=args.force)

    print("Downloaded public datasets.")
    print(f"Reentries rows: {len(reentries)}")
    print(f"Land layer: {ne_paths['land']}")
    print(f"Countries layer: {ne_paths['countries']}")


if __name__ == "__main__":
    main()
