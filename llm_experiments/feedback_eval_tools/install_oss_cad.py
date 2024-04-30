from pathlib import Path
import requests
from datetime import date, datetime, timezone, timedelta
from typing import Optional
from loguru import logger
import shutil
import os

from llm_experiments.util.path_helpers import make_data_dir

# This file contains tools to download and install the OSS CAD suite.
# Download link: https://github.com/YosysHQ/oss-cad-suite-build/releases


def _get_oss_cad_download_url(release_date: Optional[date]) -> str:
    """Returns the download URL for the OSS CAD suite, as released on a given date.
    If no date is provided, the current date (in UTC) is used (i.e., the latest release is used).
    """
    if release_date is None:
        # assume it takes about 10h for a release to be available
        release_date = (datetime.now(timezone.utc) - timedelta(hours=10)).date()

    return (
        "https://github.com/YosysHQ/oss-cad-suite-build/releases/"
        f"download/{release_date}/oss-cad-suite-linux-x64-{release_date.strftime('%Y%m%d')}.tgz"
    )


def _make_oss_cad_storage_dir() -> Path:
    return make_data_dir("tools/oss_cad_storage", append_date=False)


def install_oss_cad() -> None:
    """Downloads, unzips, and sets the PATH for the OSS CAD suite."""
    storage_dir = _make_oss_cad_storage_dir()
    download_url = _get_oss_cad_download_url(None)
    local_tgz_file = storage_dir / download_url.split("/")[-1]
    assert str(local_tgz_file).endswith(".tgz")

    if not local_tgz_file.is_file():
        logger.info("Downloading OSS CAD suite...")
        _download_large_file(download_url, local_tgz_file)
        logger.info(
            f"Downloaded OSS CAD suite file: {local_tgz_file}"
            f" (size: {local_tgz_file.stat().st_size:,} bytes)"
        )
    else:
        logger.info(f"Using existing OSS CAD suite file: {local_tgz_file}")

    local_extract_dir = storage_dir / local_tgz_file.stem
    if local_extract_dir.is_dir():
        # shutil.rmtree(local_extract_dir)
        # logger.info(f"Deleted existing directory: {local_extract_dir}")
        logger.info(f"Using existing extracted directory: {local_extract_dir}")
    else:
        shutil.unpack_archive(local_tgz_file, local_extract_dir)
        logger.info(f"Extracted OSS CAD suite to: {local_extract_dir}")

    bin_dir_path = local_extract_dir / "oss-cad-suite" / "bin"
    assert (
        bin_dir_path.is_dir()
    ), f"Expected 'bin' directory not found after extraction: {bin_dir_path}"

    # From guide: export PATH="<extracted_location>/oss-cad-suite/bin:$PATH"
    os.environ["PATH"] = f"{bin_dir_path.absolute()}:{os.environ['PATH']}"

    logger.info(f"Added OSS CAD suite to PATH: {bin_dir_path}")


def _download_large_file(url: str, path: Path) -> None:
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == "__main__":
    install_oss_cad()
