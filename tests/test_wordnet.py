import unittest
import sys
import numpy as np
from textaugment.wordnet import Wordnet


class InputTestCase(unittest.TestCase):
    def setUp(self):
        self.p = 0.8
        self.data = ["I", "am", "testing"]
        self.w = Wordnet(p=self.p)

    def test_geometric(self):
        with self.assertRaises(TypeError, msg="Receives one parameter"):
            self.w.geometric(p=self.p, data=self.data)

        with self.assertRaises(TypeError, msg="Receives one parameter"):
            self.w.geometric()

        with self.assertRaises(IndexError,  msg="Data must be set using; data='data string'"):
            self.w.geometric(data=0)

    def test_augment(self):
        with self.assertRaises(TypeError, msg="Expect string not list"):
            self.w.augment(self.data)

        with self.assertRaises(TypeError, msg="Expect string not integer"):
            self.w.augment(data=0)


class OutputTestCase(unittest.TestCase):

    def setUp(self):
        self.p = 0.8
        self.data = ["I", "am", "testing"]
        self.data2 = "известен още с псевдонимите"
        self.w = Wordnet(p=self.p)

    def test_augment(self):
        self.assertIsInstance(self.w.augment(" ".join(self.data)), str)

    def test_geometric(self):
        self.assertIsInstance(self.w.geometric(data=self.data), np.ndarray)


class PlatformTestCase(unittest.TestCase):

    def test_platform(self):
        self.assertEqual(sys.version_info[0], 3, msg="Must be using Python 3")


if __name__ == '__main__':
    unittest.main()
