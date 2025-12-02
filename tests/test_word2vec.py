import unittest
import sys
from textaugment.word2vec import Word2vec
from gensim.test.utils import common_texts
from gensim.models import Word2Vec


class InputTestCase(unittest.TestCase):

    def setUp(self):
        # create a tiny model for testing
        self.model = Word2Vec(common_texts, vector_size=20, min_count=1)
        self.wrongpath = "/tmp/non_existent_model"
        self.w = Word2vec(model=self.model)

    def test_augment(self):
        with self.assertRaises(TypeError, msg="Value for p should be float"):
            Word2vec(model=self.model, p="foo")

        with self.assertRaises(TypeError, msg="Value for runs should be integer"):
            Word2vec(model=self.model, runs="foo")

        with self.assertRaises(FileNotFoundError, msg="The model is not found"):
            Word2vec(model=self.wrongpath)

        with self.assertRaises(TypeError, msg="Input should not be lists"):
            self.w.augment(["hello"])

        with self.assertRaises(TypeError, msg="Input should not be numbers"):
            self.w.augment(45)


class OutputTestCase(unittest.TestCase):

    def setUp(self):
        self.model = Word2Vec(common_texts, vector_size=20, min_count=1)
        self.w = Word2vec(model=self.model)
        self.data = "We are testing"

    def test_augment(self):
        self.assertIsInstance(self.w.augment(self.data), str, msg="Input must be a string")
        self.assertEqual(self.w.augment("4"), "4", msg="Input should not be numbers")


class PlatformTestCase(unittest.TestCase):

    def test_platform(self):
        self.assertEqual(sys.version_info[0], 3, msg="Must be using Python 3")


if __name__ == '__main__':
    unittest.main()

