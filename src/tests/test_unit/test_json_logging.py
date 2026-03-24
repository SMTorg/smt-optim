import os
import json
import unittest
import tempfile
from unittest.mock import patch


# --- Dummy state ---
class DummyState:
    def __init__(self):
        self.iter_log = {
            "a": 1,
            "b": 2.5
        }


class DummyConfig:
    def __init__(self, results_dir):
        self.results_dir = results_dir


# --- Tests ---
class TestJsonLogger(unittest.TestCase):

    def setUp(self):
        from smt_optim.utils.logger import JsonLogger

        self.tmpdir = tempfile.TemporaryDirectory()
        self.config = DummyConfig(self.tmpdir.name)
        self.logger = JsonLogger(self.config)
        self.state = DummyState()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _get_log_path(self):
        return os.path.join(self.tmpdir.name, "stats.jsonl")

    def test_file_created_and_written(self):
        self.logger.on_iter_end(self.state)

        path = self._get_log_path()
        self.assertTrue(os.path.exists(path))

        with open(path, "r") as f:
            content = f.read()

        self.assertIn('"a": 1', content)
        self.assertIn('"b": 2.5', content)

    def test_appends_not_overwrites(self):
        self.logger.on_iter_end(self.state)
        self.logger.on_iter_end(self.state)

        path = self._get_log_path()

        with open(path, "r") as f:
            content = f.read()

        # Expect duplicated content
        self.assertGreater(len(content), 0)

        # crude but effective: key appears twice
        self.assertGreaterEqual(content.count('"a": 1'), 2)

    @patch("smt_optim.utils.json.json_safe")
    def test_json_safe_is_called(self, mock_json_safe):
        mock_json_safe.return_value = {"safe": True}

        self.logger.on_iter_end(self.state)

        self.assertGreaterEqual(mock_json_safe.call_count, 1)

        path = self._get_log_path()
        with open(path, "r") as f:
            content = f.read()

        self.assertIn('"safe": true', content)

    def test_directory_created(self):
        # Remove directory manually
        os.rmdir(self.tmpdir.name)

        self.logger.on_iter_end(self.state)

        self.assertTrue(os.path.exists(self.tmpdir.name))


if __name__ == "__main__":
    unittest.main()