import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    if path.suffix in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix in {".jsonl", ".json"}:
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")

