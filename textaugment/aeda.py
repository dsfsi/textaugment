#!/usr/bin/env python
# TextAugment: AEDA
#
# Copyright (C) 2023
# Author: Juhwan Choi
#
# URL: <https://github.com/dsfsi/textaugment/>
# For license information, see LICENSE
#
"""
This module is an implementation of the original AEDA algorithm (2021) [1].
"""
import random


class AEDA:
    """
    This class is an implementation of the original AEDA algorithm (2021) [1].

    [1] Karimi et al., 2021, November. AEDA: An Easier Data Augmentation Technique for Text Classification.
    In Findings of the Association for Computational Linguistics: EMNLP 2021 (pp. 2748-2754).
    https://aclanthology.org/2021.findings-emnlp.234.pdf

    Example usage: ::
        >>> from textaugment import AEDA
        >>> t = AEDA()
        >>> t.punct_insertion("John is going to town")
        ! John is going to town
    """

    @staticmethod
    def validate(**kwargs):
        """Validate input data"""
        if 'sentence' in kwargs:
            if not isinstance(kwargs['sentence'].strip(), str) or len(kwargs['sentence'].strip()) == 0:
                raise TypeError("sentence must be a valid sentence")

    def __init__(self, punctuations=['.', ';', '?', ':', '!', ','], random_state=1):
        """A method to initialize parameters

        :type punctuations: list
        :param punctuations: (optional) Punctuations to be inserted
        :type random_state: int
        :param random_state: (optional) Seed

        :rtype:   None
        :return:  Constructer do not return.
        """
        self.punctuations = punctuations
        self.random_state = random_state
        if isinstance(self.random_state, int):
            random.seed(self.random_state)
        else:
            raise TypeError("random_state must have type int")

    def punct_insertion(self, sentence: str):
        """Insert random punctuations to the sentence

        :type sentence: str
        :param sentence: Sentence

        :rtype:   str
        :return:  Augmented sentence
        """
        self.validate(sentence=sentence)

        sentence = sentence.strip().split(' ')
        len_sentence = len(sentence)
        # Get random number of punctuations to be inserted
        # The number of punctuations to be inserted is between 1 and 1/3 of the length of the sentence
        num_punctuations = random.randint(1, len_sentence // 3)
        augmented_sentence = sentence.copy()

        # Insert random punctuations in random positions
        for _ in range(num_punctuations):
            punct = random.choice(self.punctuations) # Select punctuation to be inserted
            pos = random.randint(0, len(augmented_sentence) - 1) # Select position to insert punctuation
            augmented_sentence = augmented_sentence[:pos] + [punct] + augmented_sentence[pos:] # Insert punctuation
        augmented_sentence = ' '.join(augmented_sentence)

        return augmented_sentence
