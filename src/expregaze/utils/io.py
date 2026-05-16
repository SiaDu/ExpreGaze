from pathlib import Path
from typing import Any, Dict, Iterable, List

import json
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(file_path: str | Path) -> Path:
    """
    Create the parent directory of a file path.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def read_json(path: str | Path) -> Any:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    path = ensure_parent_dir(path)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = ensure_parent_dir(path)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = ensure_parent_dir(path)
    df.to_csv(path, index=index)