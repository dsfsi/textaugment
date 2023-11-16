#!/usr/bin/env python
# WordNet-based data augmentation 
#
# Copyright (C) 2023
# Author: Joseph Sefara
# URL: <https://github.com/dsfsi/textaugment/>
# For license information, see LICENSE

import numpy as np
import nltk
from itertools import chain
from nltk.corpus import wordnet


class Wordnet:
    """
    A set of functions used to augment data.

    Typical usage: ::
        >>> import nltk
        >>> nltk.download('punkt')
        >>> nltk.download('wordnet')
        >>> nltk.download('averaged_perceptron_tagger')
        >>> from textaugment import Wordnet
        >>> t = Wordnet(v=True,n=True,p=0.5)
        >>> t.augment('I love school')
        i adore school
    """

    def __init__(self, **kwargs):
        """
        A method to initialize parameters

        :type random_state: int
        :param random_state: seed
        :type v: bool
        :param v: Verb, default is True
        :type n: bool
        :param n: Noun
        :type runs: int
        :param runs: Number of repetition on single text
        :type p: float, optional
        :param p: The probability of success of an individual trial. (0.1<p<1.0), default is 0.5
        :rtype:   None
        :return:  Constructer do not return.
        """

        # Set random state
        if 'random_state' in kwargs:
            self.random_state = kwargs['random_state']
            if isinstance(self.random_state, int):
                np.random.seed(self.random_state)
            else:
                raise TypeError("random_state must have type int, float, str, bytes, or bytearray")

        # Set verb to be default if no values given
        try:
            if "v" not in kwargs and "n" not in kwargs:
                kwargs['v'] = True
                kwargs['n'] = False
            elif "v" in kwargs and "n" not in kwargs:
                kwargs['v'] = True
                kwargs['n'] = False
            elif "v" not in kwargs and "n" in kwargs:
                kwargs['n'] = True
                kwargs['v'] = False
            if "runs" not in kwargs:
                kwargs['runs'] = 1

        except KeyError:
            raise

        try:
            if "p" in kwargs:
                if type(kwargs['p']) is not float:
                    raise TypeError("p represent probability of success and must be a float from 0.1 to 0.9. E.g p=0.5")
                elif type(kwargs['p']) is float:
                    self.p = kwargs['p']
            else:
                kwargs['p'] = 0.5  # Set default value
        except KeyError:
            raise

        self.p = kwargs['p']
        self.v = kwargs['v']
        self.n = kwargs['n']
        self.runs = kwargs['runs']

    def geometric(self, data):
        """
        Used to generate Geometric distribution.
        
        :type data: list
        :param data: Input data
        :rtype:   ndarray or scalar
        :return:  Drawn samples from the parameterized Geometric distribution.
        """

        data = np.array(data)
        first_trial = np.random.geometric(p=self.p, size=data.shape[0]) == 1  # Capture success after first trial
        return data[first_trial]

    def replace(self, data, lang, top_n):
        """
        The method to replace words with synonyms
        
        :type data: str
        :param data: sentence used for data augmentation
        :rtype:   str
        :return:  The augmented data
        :type lang: str
        :param lang: choose lang
        :type top_n: int
        :param top_n: top_n of synonyms to randomly choose from

        :rtype:   str
        :return:  The augmented data
        """
        data = data.lower().split()
        data_tokens = [[i, x, y] for i, (x, y) in enumerate(nltk.pos_tag(data))]  # Convert tuple to list
        if self.v:
            for loop in range(self.runs):
                words = [[i, x] for i, x, y in data_tokens if y[0] == 'V']
                words = [i for i in self.geometric(data=words)]  # List of selected words
                if len(words) >= 1:  # There are synonyms
                    for word in words:
                        synonyms1 = wordnet.synsets(word[1], wordnet.VERB, lang=lang)  # Return verbs only
                        synonyms = list(set(chain.from_iterable([syn.lemma_names(lang=lang) for syn in synonyms1])))
                        synonyms_ = []  # Synonyms with no underscores goes here
                        for w in synonyms:
                            if '_' not in w:
                                synonyms_.append(w)  # Remove words with underscores
                        if len(synonyms_) >= 1:
                            synonyms_ = synonyms_[:top_n if top_n else len(synonyms_)]  # use top n or all synonyms
                            synonym = self.geometric(data=synonyms_).tolist()
                            if synonym:  # There is a synonym
                                data[int(word[0])] = synonym[0].lower()  # Take the first success

        if self.n:
            for loop in range(self.runs):
                words = [[i, x] for i, x, y in data_tokens if y[0] == 'N']
                words = [i for i in self.geometric(data=words)]  # List of selected words
                if len(words) >= 1:  # There are synonyms
                    for word in words:
                        synonyms1 = wordnet.synsets(word[1], wordnet.NOUN, lang=lang)  # Return nouns only
                        synonyms = list(set(chain.from_iterable([syn.lemma_names(lang=lang) for syn in synonyms1])))
                        synonyms_ = []  # Synonyms with no underscores goes here
                        for w in synonyms:
                            if '_' not in w:
                                synonyms_.append(w)  # Remove words with underscores
                        if len(synonyms_) >= 1:
                            synonyms_ = synonyms_[:top_n if top_n else len(synonyms_)]  # use top n or all synonyms
                            synonym = self.geometric(data=synonyms_).tolist()
                            if synonym:  # There is a synonym
                                data[int(word[0])] = synonym[0].lower()  # Take the first success

        return " ".join(data)

    def augment(self, data, lang="eng", top_n=10):
        """
        Data augmentation for text. Generate new dataset based on verb/nouns synonyms.
        
        :type data: str
        :param data: sentence used for data augmentation 
        :rtype:   str
        :return:  The augmented data
        :type lang: str
        :param lang: choose lang
        :type top_n: int
        :param top_n: top_n of synonyms to randomly choose from

        :rtype:   str
        :return:  The augmented data
        """
        # Error handling
        if type(data) is not str:
            raise TypeError("Only strings are supported")
        if type(lang) is not str:
            raise TypeError("Only strings are supported")
        if type(top_n) is not int:
            raise TypeError("Only integers are supported")

        data = self.replace(data, lang, top_n)
        return data 
