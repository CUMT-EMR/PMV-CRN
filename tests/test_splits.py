import hashlib
import unittest

try:
    import numpy as np
    import pandas
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy and pandas are required for numerical checks")
class SplitChecks(unittest.TestCase):
    def test_blocked_split_regression(self):
        from pmv_crn.data.splits import split_indices

        labels = np.tile(np.repeat([0, 1], 32), 6)
        splits = split_indices(labels, 8, 0.6, 0.2, 2026, 32, 1, "blocked_random_stratified", 32)
        expected = [
            (185, "bdf46308ca918536b1d5cc292a274159915a71866b8bfc827b947ed50fc8eed5"),
            (64, "f2515cc2ec3a82c6741d53c4922a8d6004e03dbffca84d3d4dd16b60154b96b9"),
            (64, "0f7efa9adc15de2acb2f65b84a42dc83811811ec610daec60e14e63d609f6f19"),
        ]
        for actual, (length, digest) in zip(splits, expected):
            self.assertEqual(len(actual), length)
            self.assertEqual(hashlib.sha256(np.sort(actual).astype("<i8").tobytes()).hexdigest(), digest)

    def test_zero_purge_preserves_indices(self):
        from pmv_crn.data.splits import purge_near_splits

        splits = (np.array([0, 4, 8]), np.array([1, 5]), np.array([2, 6]))
        actual = purge_near_splits(splits, 0)
        for before, after in zip(splits, actual):
            np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
