"""
Download the datasets used by the examples.

The datasets are hosted as release archives of the `smt-optim-data <https://github.com/oihanc/smt-optim-data>`_
repository, one zip file per dataset. :func:`download_dataset` downloads and extracts one of them in a local cache,
and returns its path::

    >>> from smt_optim.utils.datasets import download_dataset
    >>> path = download_dataset("postprocessing")
    >>> data, report = load_data(path / "unconstrained")

Datasets are downloaded once: later calls reuse the cached files, so the examples also work offline afterwards.
"""

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DATA_BASE_URL = "https://github.com/oihanc/smt-optim-data/releases/download"

# Release of smt-optim-data used by default. A release never changes, which keeps the examples reproducible.
DATA_VERSION = "v0.0.1-rc1"

CACHE_ENV_VAR = "SMT_OPTIM_DATA_DIR"


def get_cache_dir() -> Path:
    """
    Directory where the datasets are stored.

    This is the ``SMT_OPTIM_DATA_DIR`` environment variable if it is set, ``~/.cache/smt-optim/datasets``
    otherwise.
    """
    env = os.environ.get(CACHE_ENV_VAR)
    return Path(env).expanduser() if env else Path.home() / ".cache" / "smt-optim" / "datasets"


def _download(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "smt-optim"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _expected_checksum(base: str, archive_name: str, timeout: float) -> str:
    """SHA-256 of `archive_name`, read from the ``checksums.txt`` file of the release."""
    text = _download(f"{base}/checksums.txt", timeout).decode()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == archive_name:
            return parts[0]
    raise ValueError(f"'{archive_name}' not found in the checksums.txt file of the release.")


def download_dataset(
    name: str,
    version: str | None = None,
    cache_dir: str | os.PathLike | None = None,
    force: bool = False,
    verify: bool = True,
    base_url: str = DATA_BASE_URL,
    timeout: float = 60.0,
    verbose: bool = True,
) -> Path:
    """
    Download a dataset of smt-optim-data and extract it in a local cache.

    Parameters
    ----------
    name : str
        Name of the dataset, e.g. ``"postprocessing"``. It is the name of a directory of the ``datasets/`` folder of
        the smt-optim-data repository.
    version : str, optional
        Release of smt-optim-data, e.g. ``"v0.1.0"``. The default is the release pinned by this version of
        smt-optim (``DATA_VERSION``).
    cache_dir : str or path-like, optional
        Directory where the datasets are stored, see :func:`get_cache_dir`. The dataset is extracted in
        ``<cache_dir>/<version>/<name>``.
    force : bool, default=False
        If True, download the dataset again even if it is already in the cache.
    verify : bool, default=True
        If True, check the SHA-256 of the archive against the ``checksums.txt`` file of the release.
    base_url : str
        Base URL of the releases. The archive is read from ``<base_url>/<version>/<name>-<version>.zip``.
    timeout : float, default=60.0
        Timeout, in seconds, of each request.
    verbose : bool, default=True
        If True, print a message when a download starts.

    Returns
    -------
    pathlib.Path
        Directory of the dataset, ``<cache_dir>/<version>/<name>``.

    Raises
    ------
    ValueError
        If the archive does not match the checksum of the release, or if it does not contain the dataset.
    RuntimeError
        If the dataset cannot be downloaded: unknown dataset or version, or no network access and the dataset is
        not in the cache.

    Examples
    --------
    >>> path = download_dataset("postprocessing")
    >>> data, report = load_data(path / "unconstrained")
    """

    version = DATA_VERSION if version is None else version
    cache_dir = get_cache_dir() if cache_dir is None else Path(cache_dir).expanduser()

    dataset_dir = cache_dir / version / name
    if dataset_dir.is_dir() and not force:
        return dataset_dir

    base = f"{base_url.rstrip('/')}/{version}"
    archive_name = f"{name}-{version}.zip"
    url = f"{base}/{archive_name}"

    if verbose:
        print(f"Downloading dataset '{name}' ({version}) from {url}")

    try:
        content = _download(url, timeout)
        expected = _expected_checksum(base, archive_name, timeout) if verify else None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"Dataset '{name}' not found in release '{version}' of smt-optim-data ({url})."
            ) from e
        raise RuntimeError(f"Could not download '{url}': HTTP error {e.code}.") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(
            f"Could not download '{url}': {e}. Check the network connection, or place the dataset in "
            f"'{dataset_dir}' manually."
        ) from e

    if expected is not None:
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected:
            raise ValueError(f"Checksum mismatch for '{archive_name}': expected {expected}, got {digest}.")

    # extract in a temporary directory first, so that an interrupted download never leaves a partial dataset
    cache_dir.joinpath(version).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir / version) as tmp:
        archive = Path(tmp) / archive_name
        archive.write_bytes(content)

        extracted = Path(tmp) / "extracted"
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)

        if not (extracted / name).is_dir():
            raise ValueError(f"Archive '{archive_name}' does not contain a '{name}/' directory.")

        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        shutil.move(str(extracted / name), str(dataset_dir))

    return dataset_dir
