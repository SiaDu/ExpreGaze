from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def unpack_if_needed(files_dir: Path, zip_name: str, target_dir_name: str) -> Path:
    zip_path = files_dir / zip_name
    target_dir = files_dir / target_dir_name

    if target_dir.exists():
        print(f"skip: {target_dir} already exists")
        return target_dir

    if not zip_path.exists():
        raise FileNotFoundError(f"Missing zip: {zip_path}")

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    print(f"unpacked: {zip_path} -> {target_dir}")
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files-dir", required=True)
    args = parser.parse_args()

    files_dir = Path(args.files_dir).expanduser().resolve()

    unpack_if_needed(files_dir, "annotation.v1.zip", "annotation.v1")
    unpack_if_needed(files_dir, "meta.v1.zip", "meta.v1")
    unpack_if_needed(files_dir, "script1K.v1.zip", "script1K.v1")
    unpack_if_needed(files_dir, "subtitle1K.v1.zip", "subtitle1K.v1")


if __name__ == "__main__":
    main()
