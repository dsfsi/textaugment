import unittest
import sys
import numpy as np
from textaugment import MIXUP


class OutputTestCase(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(12).reshape(4, 3).astype(float)
        self.y = np.arange(8).reshape(4, 2).astype(float)

    def test_flow_accepts_list_of_lists(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        labels = [[1.0, 0.0], [0.0, 1.0]]
        generator, num_batches = MIXUP(random_state=1).flow(data, labels=labels, batch_size=2)
        mixed_x, mixed_y = next(generator)
        self.assertEqual(np.asarray(mixed_x).shape, (2, 2))
        self.assertEqual(np.asarray(mixed_y).shape, (2, 2))
        self.assertEqual(num_batches, 1)

    def test_flow_accepts_arrays(self):
        generator, num_batches = MIXUP(random_state=1).flow(self.x, labels=self.y, batch_size=2)
        mixed_x, mixed_y = next(generator)
        self.assertEqual(np.asarray(mixed_x).shape, (2, 3))
        self.assertEqual(np.asarray(mixed_y).shape, (2, 2))
        self.assertEqual(num_batches, 2)


class PlatformTestCase(unittest.TestCase):

    def test_platform(self):
        self.assertEqual(sys.version_info[0], 3, msg="Must be using Python 3")


if __name__ == '__main__':
    unittest.main()
