import unittest
import numpy as np
import warnings
import json

# Adjust this import to your actual module
from smtoptim.utils.json import json_safe


class TestJsonSafe(unittest.TestCase):

    # test primitives
    def test_none(self):
        self.assertIsNone(json_safe(None))

    def test_primitives(self):
        self.assertEqual(json_safe(True), True)
        self.assertEqual(json_safe(5), 5)
        self.assertEqual(json_safe(3.14), 3.14)
        self.assertEqual(json_safe("hello"), "hello")


    # numpy scalars
    def test_numpy_integer(self):
        x = np.int64(42)
        result = json_safe(x)
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)

    def test_numpy_floating(self):
        x = np.float64(2.5)
        result = json_safe(x)
        self.assertEqual(result, 2.5)
        self.assertIsInstance(result, float)


    # numpy array
    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        self.assertEqual(json_safe(arr), [1, 2, 3])


    # dictionary
    def test_dict_simple(self):
        d = {"a": 1, "b": np.int64(3)}
        result = json_safe(d)
        self.assertEqual(result, {"a": 1, "b": 3})

    def test_dict_key_string_conversion(self):
        d = {1: "a", 2.5: "b"}
        result = json_safe(d)
        self.assertEqual(result, {"1": "a", "2.5": "b"})

    def test_dict_with_unserializable_value(self):
        class Bad:
            def __repr__(self):
                return "BadObject"

        d = {"a": Bad()}
        result = json_safe(d)
        self.assertEqual(result, {"a": None})


    # list and tuple
    def test_list(self):
        data = [1, np.int64(2), np.array([3, 4])]
        result = json_safe(data)
        self.assertEqual(result, [1, 2, [3, 4]])

    def test_tuple(self):
        data = (1, np.float64(2.5))
        result = json_safe(data)
        self.assertEqual(result, [1, 2.5])  # tuple becomes list

    def test_list_with_unserializable(self):
        class Bad:
            pass

        data = [1, Bad()]
        result = json_safe(data)
        self.assertEqual(result, [1, None])


    # already JSON serializable
    def test_json_serializable_object(self):
        data = {"x": 1, "y": [1, 2]}
        result = json_safe(data)
        self.assertEqual(result, data)


    # outer exception handling
    def test_outer_exception_returns_none_and_warns(self):
        class Exploding:
            def __repr__(self):
                raise RuntimeError("Boom")

        obj = Exploding()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = json_safe(obj)

            self.assertIsNone(result)
            self.assertTrue(len(w) > 0)
            self.assertTrue("Failed to convert" in str(w[0].message))


if __name__ == "__main__":
    unittest.main()