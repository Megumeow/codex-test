Place an optional population raster here if you want raster-based exposure.

Supported local files:

- `*.tif`
- `*.tiff`
- `*.vrt`

Typical use:

1. Put a population count raster in this folder.
2. Install `rasterio`.
3. Run `python scripts/run_demo.py`.

If no raster is present, the project falls back to country-level population exposure automatically.
