"""Preprocess cortex.parquet into a snappy-compressed optimized file.

Run once before starting the app:
    uv run python scripts/preprocess_data.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "cortex.parquet"
DST = ROOT / "cortex_optimized.parquet"


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Source file not found: {SRC}")

    print(f"Reading {SRC} ...")
    lf = pl.scan_parquet(SRC)

    schema = lf.collect_schema()
    print(f"Schema: {schema}")

    df = lf.collect()
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns}")
    print(f"\nSample:\n{df.head(5)}")

    print(f"\nBuildings in data:")
    buildings = (
        df.filter(pl.col("property_name").is_not_null())
        .select("property_name")
        .unique()
        .sort("property_name")
    )
    print(buildings)

    print(f"\nWriting optimized parquet to {DST} ...")
    df.write_parquet(DST, compression="snappy")
    print(f"Done. Size: {DST.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
