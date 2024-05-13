from pathlib import Path
from typing import Literal

import polars as pl
from loguru import logger


def merge_jsonl_to_parquet(jsonl_path: Path, ext: Literal[".pq", ".parquet"] = ".pq") -> None:
    """Merges the JSONL file into a Parquet file, at the same place as the .jsonl file."""
    # merge the .jsonl file into a parquet file
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    dest_pq_file_path = jsonl_path.with_suffix(ext)

    df = pl.read_ndjson(jsonl_path)
    df.write_parquet(dest_pq_file_path)
    logger.info(f"Saved experiment data from JSONL to Parquet: {len(df):,} rows.")
