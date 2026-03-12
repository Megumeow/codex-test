Drop optional historical GP / TLE files here.

Supported formats:

- CSV with Space-Track style columns such as `NORAD_CAT_ID`, `EPOCH`, `MEAN_MOTION`, `BSTAR`
- JSON arrays of Space-Track style records
- TLE history text files with repeated line 1 / line 2 pairs

If a filename contains a NORAD ID like `25544_history.txt`, the loader will use it as a fallback object identifier.
