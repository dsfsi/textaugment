import unittest
import sys
from textaugment import AEDA


class OutputTestCase(unittest.TestCase):

    def setUp(self):
        self.t = AEDA(random_state=1)

    def test_punct_insertion(self):
        self.assertIsInstance(self.t.punct_insertion("john is going to town"), str)

    def test_punct_insertion_short_sentence(self):
        for sentence in ("good", "very good"):
            augmented = self.t.punct_insertion(sentence)
            self.assertIsInstance(augmented, str)
            self.assertEqual(len(augmented.split()), len(sentence.split()) + 1)


class PlatformTestCase(unittest.TestCase):

    def test_platform(self):
        self.assertEqual(sys.version_info[0], 3, msg="Must be using Python 3")


if __name__ == '__main__':
    unittest.main()
