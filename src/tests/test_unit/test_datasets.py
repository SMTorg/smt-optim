import hashlib
import io
import os
import shutil
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock

from smt_optim.utils import datasets
from smt_optim.utils.datasets import download_dataset, get_cache_dir


def _make_release(root, version="v1.0.0", name="demo", files=None, checksum=None):
    """Create a local release ``<root>/<version>/`` with one dataset archive and its checksums.txt."""
    files = (
        {"a/doe.csv": "iter,f0\n0,1.0\n", "b.txt": "hello"} if files is None else files
    )
    release = Path(root) / version
    release.mkdir(parents=True)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for path, content in files.items():
            zf.writestr(f"{name}/{path}", content)
    content = buffer.getvalue()

    archive_name = f"{name}-{version}.zip"
    (release / archive_name).write_bytes(content)
    digest = hashlib.sha256(content).hexdigest() if checksum is None else checksum
    (release / "checksums.txt").write_text(f"{digest}  {archive_name}\n")
    return release


class TestDownloadDataset(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.remote = self.tmp / "remote"
        self.cache = self.tmp / "cache"
        self.base_url = self.remote.as_uri()

    def _download(self, name="demo", version="v1.0.0", **kwargs):
        return download_dataset(
            name,
            version=version,
            cache_dir=self.cache,
            base_url=self.base_url,
            verbose=False,
            **kwargs,
        )

    def test_downloads_and_extracts_the_dataset(self):
        _make_release(self.remote)
        path = self._download()

        self.assertEqual(path, self.cache / "v1.0.0" / "demo")
        self.assertEqual((path / "a" / "doe.csv").read_text(), "iter,f0\n0,1.0\n")
        self.assertEqual((path / "b.txt").read_text(), "hello")

    def test_no_temporary_file_is_left_in_the_cache(self):
        _make_release(self.remote)
        self._download()

        self.assertEqual([p.name for p in (self.cache / "v1.0.0").iterdir()], ["demo"])

    def test_cached_dataset_is_not_downloaded_again(self):
        release = _make_release(self.remote)
        first = self._download()
        shutil.rmtree(release)  # the remote is not reachable anymore

        self.assertEqual(self._download(), first)

    def test_force_downloads_again(self):
        _make_release(self.remote)
        path = self._download()
        (path / "b.txt").write_text("modified")
        (path / "extra.txt").write_text("x")

        self._download(force=True)

        self.assertEqual((path / "b.txt").read_text(), "hello")
        self.assertFalse((path / "extra.txt").exists())

    def test_datasets_are_independent(self):
        _make_release(self.remote, name="demo")
        shutil.copy(
            self.remote / "v1.0.0" / "demo-v1.0.0.zip",
            self.remote / "v1.0.0" / "other-v1.0.0.zip",
        )
        # the copy is a "demo" archive: not a valid "other" dataset
        with open(self.remote / "v1.0.0" / "checksums.txt", "a") as f:
            f.write(
                hashlib.sha256(
                    (self.remote / "v1.0.0" / "other-v1.0.0.zip").read_bytes()
                ).hexdigest()
                + "  other-v1.0.0.zip\n"
            )

        self.assertTrue(self._download(name="demo").is_dir())
        with self.assertRaises(ValueError):
            self._download(name="other")
        self.assertFalse((self.cache / "v1.0.0" / "other").exists())

    def test_checksum_mismatch_raises_and_nothing_is_cached(self):
        _make_release(self.remote, checksum="0" * 64)

        with self.assertRaises(ValueError):
            self._download()
        self.assertFalse((self.cache / "v1.0.0" / "demo").exists())

    def test_verify_false_skips_the_checksum(self):
        _make_release(self.remote, checksum="0" * 64)

        self.assertTrue(self._download(verify=False).is_dir())

    def test_archive_missing_from_checksums_raises(self):
        release = _make_release(self.remote)
        (release / "checksums.txt").write_text("")

        with self.assertRaises(ValueError):
            self._download()

    def test_archive_without_the_dataset_directory_raises(self):
        release = _make_release(self.remote, name="wrong")
        shutil.move(release / "wrong-v1.0.0.zip", release / "demo-v1.0.0.zip")

        with self.assertRaises(ValueError):
            self._download(verify=False)

    def test_unreachable_release_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            self._download(version="v9.9.9")

    def test_unknown_dataset_http_404(self):
        error = urllib.error.HTTPError("url", 404, "Not Found", None, None)
        with mock.patch.object(datasets, "_download", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "not found"):
                self._download()

    def test_default_version_is_used(self):
        _make_release(self.remote, version=datasets.DATA_VERSION)
        path = download_dataset(
            "demo", cache_dir=self.cache, base_url=self.base_url, verbose=False
        )

        self.assertEqual(path, self.cache / datasets.DATA_VERSION / "demo")

    def test_verbose_prints_only_when_downloading(self):
        _make_release(self.remote)
        with mock.patch("builtins.print") as printed:
            download_dataset(
                "demo", version="v1.0.0", cache_dir=self.cache, base_url=self.base_url
            )
            self.assertEqual(printed.call_count, 1)
            download_dataset(
                "demo", version="v1.0.0", cache_dir=self.cache, base_url=self.base_url
            )
            self.assertEqual(printed.call_count, 1)


class TestCacheDir(unittest.TestCase):
    def test_default(self):
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop(datasets.CACHE_ENV_VAR, None)
            self.assertEqual(
                get_cache_dir(), Path.home() / ".cache" / "smt-optim" / "datasets"
            )

    def test_environment_variable(self):
        with mock.patch.dict(os.environ, {datasets.CACHE_ENV_VAR: "/some/where"}):
            self.assertEqual(get_cache_dir(), Path("/some/where"))

    def test_environment_variable_is_used_by_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote = Path(tmp) / "remote"
            _make_release(remote)
            with mock.patch.dict(
                os.environ, {datasets.CACHE_ENV_VAR: str(Path(tmp) / "env_cache")}
            ):
                path = download_dataset(
                    "demo", version="v1.0.0", base_url=remote.as_uri(), verbose=False
                )

            self.assertEqual(path, Path(tmp) / "env_cache" / "v1.0.0" / "demo")


if __name__ == "__main__":
    unittest.main()
