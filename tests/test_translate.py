import unittest
import sys
from textaugment.translate import Translate
from textaugment import translate


class InputTestCase(unittest.TestCase):

    def setUp(self):
        self.t = Translate(src="en", to="es")

    def test_geometric(self):
        with self.assertRaises(ValueError, msg="Parameters nust be set"):
            Translate()

        with self.assertRaises(KeyError, msg="Value of parameters must be correct"):
            Translate(to=7, src="hello")  # Test parameter, type

        with self.assertRaises(TypeError, msg="Only strings are allowed"):
            self.t.augment(45)

    def test_translate(self):
        self.assertTrue(translate.LANGUAGES, msg="Files exists")


class OutputTestCase(unittest.TestCase):

    def setUp(self):
        self.t = Translate(src="en", to="es")
        self.data = "He walks"

    def test_augment(self):
        self.assertEqual(self.t.augment(self.data), self.data)

        self.assertEqual(self.t.augment("4"), "4")


class PlatformTestCase(unittest.TestCase):

    def test_platform(self):
        self.assertEqual(sys.version_info[0], 3, msg="Must be using Python 3")


if __name__ == '__main__':
    unittest.main()

