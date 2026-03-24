import unittest
from io import StringIO
from unittest.mock import patch


# --- Minimal stubs ---
class DummySample:
    def __init__(self, obj):
        self.obj = obj


class DummyState:
    def __init__(self, iter_, budget):
        self.iter = iter_
        self.budget = budget
        self.iter_log = {
            "fidelity": 2,
            "gp_training_time": 0.123,
            "acq_opt_time": 0.456,
        }

    def get_best_sample(self, ctol):
        return DummySample([1.2345])


class DummyConfig:
    pass


# --- Tests ---
class TestConsoleLogger(unittest.TestCase):

    def setUp(self):
        from smt_optim.utils.logger import ConsoleLogger

        self.logger = ConsoleLogger(DummyConfig())
        self.state = DummyState(iter_=0, budget=1.0)

    @patch("sys.stdout", new_callable=StringIO)
    def test_header_printed_on_first_iteration(self, mock_stdout):
        self.logger.on_iter_end(self.state)

        output = mock_stdout.getvalue().strip().split("\n")

        # First line should be header
        self.assertIn("iter", output[0])
        self.assertIn("budget", output[0])

        # Second line should be data row
        self.assertEqual(len(output), 2)

    @patch("sys.stdout", new_callable=StringIO)
    def test_no_header_before_repeat(self, mock_stdout):
        # First call prints header
        self.logger.on_iter_end(self.state)

        # Reset capture
        mock_stdout.truncate(0)
        mock_stdout.seek(0)

        # Next call should NOT print header
        self.state.iter = 1
        self.logger.on_iter_end(self.state)

        output = mock_stdout.getvalue().strip().split("\n")

        self.assertEqual(len(output), 1)  # only row, no header

    @patch("sys.stdout", new_callable=StringIO)
    def test_header_repeats(self, mock_stdout):
        # Force repeat condition
        self.logger.iter = self.logger.repeat_header

        self.logger.on_iter_end(self.state)

        output = mock_stdout.getvalue().strip().split("\n")

        self.assertEqual(len(output), 2)
        self.assertIn("iter", output[0])  # header present

    @patch("sys.stdout", new_callable=StringIO)
    def test_row_formatting(self, mock_stdout):
        self.logger.on_iter_end(self.state)

        output = mock_stdout.getvalue().strip().split("\n")
        row = output[1]

        # Check formatted values (string-level assertions)
        self.assertIn("0", row)              # iter
        self.assertIn("1.000", row)          # budget (.3f)
        self.assertIn("1.23450e+00", row)    # fmin (.5e)
        self.assertIn("2", row)              # fidelity

    def test_iteration_counter_increments(self):
        self.assertEqual(self.logger.iter, 0)

        self.logger.on_iter_end(self.state)
        self.assertEqual(self.logger.iter, 1)

        self.logger.on_iter_end(self.state)
        self.assertEqual(self.logger.iter, 2)


if __name__ == "__main__":
    unittest.main()