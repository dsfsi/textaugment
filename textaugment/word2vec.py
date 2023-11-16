#!/usr/bin/env python
# Word2vec-based data augmentation 
#
# Copyright (C) 2023
# Author: Joseph Sefara
# URL: <https://github.com/dsfsi/textaugment/>
# For license information, see LICENSE

import gensim
import numpy as np
import random


class Word2vec:
    """
    A set of functions used to augment data.

    Typical usage: :: 
        >>> from textaugment import Word2vec
        >>> t = Word2vec(model='path/to/gensim/model'or 'gensim model itself')
        >>> t.augment('I love school', top_n=10)
        i adore school
    """
    
    def __init__(self, **kwargs):
        """
        A method to initialize a model on a given path.
        :type random_state: int, float, str, bytes, bytearray
        :param random_state: seed
        :type model: str or gensim.models.word2vec.Word2Vec or gensim.models.fasttext.FastText
        :param model: The path to the model or the model itself.
        :type runs: int, optional
        :param runs: The number of times to augment a sentence. By default is 1.
        :type v: bool or optional
        :param v: Replace all the words if true. If false randomly replace words.
                Used in a Paper (https://www.cs.cmu.edu/~diyiy/docs/emnlp_wang_2015.pdf)
        :type p: float, optional
        :param p: The probability of success of an individual trial. (0.1<p<1.0), default is 0.5
        """

        # Set random state
        if 'random_state' in kwargs:
            self.random_state = kwargs['random_state']
            if isinstance(self.random_state, int):
                random.seed(self.random_state)
                np.random.seed(self.random_state)
            else:
                raise TypeError("random_state must have type int")

        # Set verbose to false if does not exists
        try:
            if kwargs['v']: 
                self.v = True
            else:
                self.v = False
        except KeyError:
            self.v = False

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

        # Error handling of given parameters
        try:
            if "runs" not in kwargs:
                kwargs["runs"] = 1  # Default value for runs
            elif type(kwargs["runs"]) is not int:
                raise TypeError("DataType for 'runs' must be an integer")
            if "model" not in kwargs:
                raise ValueError("Set the value of model. e.g model='path/to/model' or model itself")
            if type(kwargs['model']) != str and 'gensim' not in str(type(kwargs['model'])).lower():
                raise TypeError("Model path must be a string, or the type of the model must be a gensim.models...")
        except (ValueError, TypeError):
            raise
        else:
            self.runs = kwargs["runs"] 
            self.model = kwargs["model"]
            self.p = kwargs["p"]
            try:
                if type(self.model) is str:
                    self.model = gensim.models.Word2Vec.load(self.model)  # load word2vec or fasttext model
            except FileNotFoundError:
                print("Error: Model not found. Verify the path.\n")
                raise ValueError("Error: Model not found. Verify the path.")

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

    def augment(self, data: str, top_n: int = 10):
        """
        The method to replace words with similar words.
        
        :type data: str
        :param data: Input data
        :type top_n: int
        :param top_n: top_n of most similar words to randomly choose from

        :rtype:   str
        :return:  The augmented data
        """

        # Avoid nulls and other unsupported types
        if type(top_n) is not int:
            raise TypeError("Only integers are supported")
        if type(data) is not str: 
            raise TypeError("Only strings are supported")
        # Lower case and split
        data_tokens = data.lower().split()

        # Verbose = True then replace all the words.
        if self.v:
            for _ in range(self.runs):
                for index in range(len(data_tokens)):  # Index from 0 to length of data_tokens
                    try:
                        similar_words = [syn for syn, t in self.model.wv.most_similar(data_tokens[index], topn=top_n)]
                        r = random.randrange(len(similar_words))
                        data_tokens[index] = similar_words[r].lower()  # Replace with random synonym from 10 synonyms
                    except KeyError:
                        pass  # For words not in the word2vec model
        else:  # Randomly replace some words
            for _ in range(self.runs):
                data_tokens_idx = [[x, y] for (x, y) in enumerate(data_tokens)]  # Enumerate data
                words = self.geometric(data=data_tokens_idx).tolist()  # List of words indexed
                for w in words:
                    try:
                        similar_words_and_weights = [(syn, t) for syn, t in self.model.wv.most_similar(w[1])]
                        similar_words = [word for word, t in similar_words_and_weights]
                        similar_words_weights = [t for word, t in similar_words_and_weights]
                        word = random.choices(similar_words, similar_words_weights, k=1)
                        data_tokens[int(w[0])] = word[0].lower()  # Replace with random synonym from 10 synonyms
                    except KeyError:
                        pass
            return " ".join(data_tokens)
        return " ".join(data_tokens)


class Fasttext(Word2vec):
    """
    A set of functions used to augment data.

    Typical usage: ::
        >>> from textaugment import Fasttext
        >>> t = Fasttext('path/to/gensim/model'or 'gensim model itself')
        >>> t.augment('I love school', top_n=10)
        i adore school
    """
    pass
