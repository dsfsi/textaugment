#!/usr/bin/env python
# TextAugment: EDA
#
# Copyright (C) 2018-2020
# Author: Joseph Sefara
#
# URL: <https://github.com/dsfsi/textaugment/>
# For license information, see LICENSE
#
"""
This module is an implementation of the original EDA algorithm (2019) [1].
"""
from nltk.corpus import wordnet, stopwords
import random


class EDA:
    """
    This class is an implementation of the original EDA algorithm (2019) [1].

    [1] Wei, J. and Zou, K., 2019, November. EDA: Easy Data Augmentation Techniques for Boosting Performance on
    Text Classification Tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing
    and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (pp. 6383-6389).
    https://www.aclweb.org/anthology/D19-1670.pdf

    Example usage: ::
        >>> from textaugment import EDA
        >>> t = EDA()
        >>> t.synonym_replacement("John is going to town")
        John is give out to town
        >>> t.random_deletion("John is going to town", p=0.2)
        is going to town
        >>> t.random_swap("John is going to town")
        John town going to is
        >>> t.random_insertion("John is going to town")
        John is going to make up town
    """

    @staticmethod
    def _get_synonyms(word):
        """Generate synonym"""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym)
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

    @staticmethod
    def swap_word(new_words):
        """Swap words"""
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        return new_words

    @staticmethod
    def validate(**kwargs):
        """Validate input data"""

        if 'p' in kwargs:
            if kwargs['p'] > 1 or kwargs['p'] < 0:
                raise TypeError("p must be a fraction between 0 and 1")
        if 'sentence' in kwargs:
            if not isinstance(kwargs['sentence'].strip(), str) or len(kwargs['sentence'].strip()) == 0:
                raise TypeError("sentence must be a valid sentence")
        if 'n' in kwargs:
            if not isinstance(kwargs['n'], int):
                raise TypeError("n must be a valid integer")

    def __init__(self, stop_words=None, random_state=1):
        """A method to initialize parameters

        :type random_state: int
        :param random_state: (optional) Seed
        :type stop_words: list
        :param stop_words: (optional) List of stopwords

        :rtype:   None
        :return:  Constructer do not return.
        """
        self.stopwords = stopwords.words('english') if stop_words is None else stop_words
        self.sentence = None
        self.p = None
        self.n = None
        self.random_state = random_state
        if isinstance(self.random_state, int):
            random.seed(self.random_state)
        else:
            raise TypeError("random_state must have type int")

    def add_word(self, new_words):
        """Insert word"""
        synonyms = list()
        counter = 0
        while len(synonyms) < 1:
            random_word_list = list([word for word in new_words if word not in self.stopwords])
            random_word = random_word_list[random.randint(0, len(random_word_list) - 1)]
            synonyms = self._get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return new_words  # See Issue 14 for details
        random_synonym = synonyms[0]  # TODO
        random_idx = random.randint(0, len(new_words) - 1)
        new_words.insert(random_idx, random_synonym)
        return new_words

    def synonym_replacement(self, sentence: str, n: int = 1):
        """Replace n words in the sentence with synonyms from wordnet

        :type sentence: str
        :param sentence: Sentence
        :type n: int
        :param n: Number of repetitions to replace

        :rtype:   str
        :return:  Augmented sentence
        """
        self.validate(sentence=sentence, n=n)
        self.n = n
        self.sentence = sentence
        words = sentence.split()
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stopwords]))
        random.shuffle(random_word_list)
        replaced = 0
        for random_word in random_word_list:
            synonyms = self._get_synonyms(random_word)
            if len(synonyms) > 0:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                replaced += 1
            if replaced >= self.n:
                break
        sentence = ' '.join(new_words)

        return sentence

    def random_deletion(self, sentence: str, p: float = 0.1):
        """Randomly delete words from the sentence with probability p

        :type sentence: str
        :param sentence: Sentence
        :type p: int
        :param p: Probability between 0 and 1

        :rtype:   str
        :return:  Augmented sentence
        """
        self.validate(sentence=sentence, p=p)
        self.p = p
        self.sentence = sentence
        words = sentence.split()
        if len(words) == 1:
            return words
        new_words = list()
        for word in words:
            r = random.uniform(0, 1)
            if r > self.p:
                new_words.append(word)
        # if all words are deleted, just return a random word
        if len(new_words) == 0:
            return random.choice(words)

        return " ".join(new_words)

    def random_swap(self, sentence: str, n: int = 1):
        """Randomly swap two words in the sentence n times

        :type sentence: str
        :param sentence: Sentence
        :type n: int
        :param n: Number of repetitions to swap

        :rtype:   str
        :return:  Augmented sentence
        """
        self.validate(sentence=sentence, n=n)
        self.n = n
        self.sentence = sentence
        words = sentence.split()
        new_words = words.copy()
        for _ in range(self.n):
            new_words = self.swap_word(new_words)
        return " ".join(new_words)

    def random_insertion(self, sentence: str, n: int = 1):
        """Randomly insert n words into the sentence

        :type sentence: str
        :param sentence: Sentence
        :type n: int
        :param n: Number of words to insert

        :rtype:   str
        :return:  Augmented sentence
        """
        self.validate(sentence=sentence, n=n)
        self.n = n
        self.sentence = sentence
        words = sentence.split()
        new_words = words.copy()
        for _ in range(self.n):
            new_words = self.add_word(new_words)
        return " ".join(new_words)
