import unittest
import sys
import numpy as np
from textaugment import MIXUP


class OutputTestCase(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(12).reshape(4, 3).astype(float)
        self.y = np.arange(8).reshape(4, 2).astype(float)

    def test_mixup_data_without_labels_and_runs(self):
        mixed = MIXUP(random_state=1, runs=3).mixup_data(self.x, y=None)
        self.assertEqual(mixed.shape[0], 3 * self.x.shape[0])

    def test_mixup_data_with_labels_and_runs(self):
        mixed_x, mixed_y = MIXUP(random_state=1, runs=3).mixup_data(self.x, y=self.y)
        self.assertEqual(mixed_x.shape[0], 3 * self.x.shape[0])
        self.assertEqual(mixed_y.shape[0], 3 * self.y.shape[0])


class PlatformTestCase(unittest.TestCase):

    def test_platform(self):
        self.assertEqual(sys.version_info[0], 3, msg="Must be using Python 3")


if __name__ == '__main__':
    unittest.main()
