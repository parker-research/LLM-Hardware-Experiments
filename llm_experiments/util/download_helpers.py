from pathlib import Path

import requests


def download_large_file(url: str, path: Path) -> None:
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
