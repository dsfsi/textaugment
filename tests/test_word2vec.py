import unittest
import sys
from textaugment.word2vec import Word2vec


class InputTestCase(unittest.TestCase):

    def setUp(self):
        self.path = "/home/tjs/dev/papu/models/gensim_cbow_sepedi"
        self.wrongpath = "/home/tjs/dev/papu/models/gensim_cbow_sepedi-wrong"
        self.w = Word2vec(model=self.path)

    def test_augment(self):
        with self.assertRaises(TypeError, msg="Value for p should be float"):
            Word2vec(model=self.path, p="foo")

        with self.assertRaises(TypeError, msg="Value for runs should be integer"):
            Word2vec(model=self.path, runs="foo")

        with self.assertRaises(FileNotFoundError, msg="The model is not found"):
            Word2vec(model=self.wrongpath)

        with self.assertRaises(TypeError, msg="Input should not be lists"):
            self.w.augment(["hello"])

        with self.assertRaises(TypeError, msg="Input should not be numbers"):
            self.w.augment(45)


class OutputTestCase(unittest.TestCase):

    def setUp(self):
        self.path = "/home/tjs/dev/papu/models/gensim_cbow_sepedi"
        self.w = Word2vec(model=self.path)
        self.data = "We are testing"

    def test_augment(self):
        self.assertIsInstance(self.w.augment(self.data), str, msg="Input must be a string")
        self.assertEqual(self.w.augment("4"), "4", msg="Input should not be numbers")


class PlatformTestCase(unittest.TestCase):

    def test_platform(self):
        self.assertEqual(sys.version_info[0], 3, msg="Must be using Python 3")


if __name__ == '__main__':
    unittest.main()

