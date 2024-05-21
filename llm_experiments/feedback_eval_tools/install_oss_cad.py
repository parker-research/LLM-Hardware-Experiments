from pathlib import Path
from datetime import date, datetime, timezone, timedelta
from typing import Optional
from loguru import logger
import shutil
import os

from llm_experiments.util.path_helpers import make_data_dir
from llm_experiments.util.download_helpers import download_large_file

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


def install_oss_cad_and_activate(release_date: Optional[date] = None) -> None:
    """Downloads, unzips, and sets the PATH for the OSS CAD suite.

    Args:
        release_date (Optional[date]): The release date of the OSS CAD suite to install.
            If None, the latest release is used.
    """
    storage_dir = _make_oss_cad_storage_dir()
    download_url = _get_oss_cad_download_url(release_date)
    local_tgz_file = storage_dir / download_url.split("/")[-1]
    assert str(local_tgz_file).endswith(".tgz")

    if not local_tgz_file.is_file():
        logger.info(f"Downloading OSS CAD suite ({release_date=})...")
        download_large_file(download_url, local_tgz_file)
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
    # FIXME: can't use multiple version of this tool in a single execution
    # FIXME: this PATH step should be moved up one level of abstraction, upon usage

    logger.info(f"Added OSS CAD suite to PATH: {bin_dir_path}")


if __name__ == "__main__":
    install_oss_cad_and_activate()
