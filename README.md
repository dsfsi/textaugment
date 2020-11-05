

# [TextAugment: Improving Short Text Classification through Global Augmentation Methods](https://arxiv.org/abs/1907.03752) 

[![licence](https://img.shields.io/github/license/dsfsi/textaugment.svg?maxAge=3600)](https://github.com/dsfsi/textaugment/blob/master/LICENCE) [![GitHub release](https://img.shields.io/github/release/dsfsi/textaugment.svg?maxAge=3600)](https://github.com/dsfsi/textaugment/releases) [![Wheel](https://img.shields.io/pypi/wheel/textaugment.svg?maxAge=3600)](https://pypi.python.org/pypi/textaugment) [![python](https://img.shields.io/pypi/pyversions/textaugment.svg?maxAge=3600)](https://pypi.org/project/textaugment/) [![TotalDownloads](https://pepy.tech/badge/textaugment)](https://pypi.org/project/textaugment/) [![Downloads](https://static.pepy.tech/badge/textaugment/month)](https://pypi.org/project/textaugment/) [![LNCS](https://img.shields.io/badge/LNCS-Book%20Chapter-B31B1B.svg)](https://link.springer.com/chapter/10.1007%2F978-3-030-57321-8_21) [![arxiv](https://img.shields.io/badge/cs.CL-arXiv%3A1907.03752-B31B1B.svg)](https://arxiv.org/abs/1907.03752)

## You have just found TextAugment.

TextAugment is a Python 3 library for augmenting text for natural language processing applications. TextAugment stands on the giant shoulders of [NLTK](https://www.nltk.org/), [Gensim](https://radimrehurek.com/gensim/), and [TextBlob](https://textblob.readthedocs.io/) and plays nicely with them.

# Table of Contents

- [Features](#Features)
- [Citation Paper](#citation-paper) 
	- [Requirements](#Requirements)
	- [Installation](#Installation)
	- [How to use](#How-to-use)
		- [Word2vec-based augmentation](#Word2vec-based-augmentation)
		- [WordNet-based augmentation](#WordNet-based-augmentation)
		- [RTT-based augmentation](#RTT-based-augmentation)
- [Easy data augmentation (EDA)](#eda-easy-data-augmentation-techniques-for-boosting-performance-on-text-classification-tasks)
- [Mixup augmentation](#mixup-augmentation)
  - [Implementation](#Implementation)
- [Acknowledgements](#Acknowledgements)

## Features

- Generate synthetic data for improving model performance without manual effort
- Simple, lightweight, easy-to-use library.
- Plug and play to any machine learning frameworks (e.g. PyTorch, TensorFlow, Scikit-learn)
- Support textual data

## Citation Paper

**[Improving short text classification through global augmentation methods](https://link.springer.com/chapter/10.1007%2F978-3-030-57321-8_21)**.



![alt text](https://raw.githubusercontent.com/dsfsi/textaugment/master/augment.png "Augmentation methods")

### Requirements

* Python 3

The following software packages are dependencies and will be installed automatically.

```shell
$ pip install numpy nltk gensim textblob googletrans 

```
The following code downloads NLTK corpus for [wordnet](http://www.nltk.org/howto/wordnet.html).
```python
nltk.download('wordnet')
```
The following code downloads [NLTK tokenizer](https://www.nltk.org/_modules/nltk/tokenize/punkt.html). This tokenizer divides a text into a list of sentences by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences. 
```python
nltk.download('punkt')
```
The following code downloads default [NLTK part-of-speech tagger](https://www.nltk.org/_modules/nltk/tag.html) model. A part-of-speech tagger processes a sequence of words, and attaches a part of speech tag to each word.
```python
nltk.download('averaged_perceptron_tagger')
```
Use gensim to load a pre-trained word2vec model. Like [Google News from Google drive](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit).
```python
import gensim
model = gensim.models.Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
```
You can also use gensim to load Facebook's Fasttext [English](https://fasttext.cc/docs/en/english-vectors.html) and [Multilingual models](https://fasttext.cc/docs/en/crawl-vectors.html)
```
import gensim
model = gensim.models.fasttext.load_facebook_model('./cc.en.300.bin.gz')
```

Or training one from scratch using your data or the following public dataset:

- [Text8 Wiki](http://mattmahoney.net/dc/enwik9.zip)

- [Dataset from "One Billion Word Language Modeling Benchmark"](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz)

### Installation

Install from pip [Recommended] 
```sh
$ pip install textaugment
or install latest release
$ pip install git+git@github.com:dsfsi/textaugment.git
```

Install from source
```sh
$ git clone git@github.com:dsfsi/textaugment.git
$ cd textaugment
$ python setup.py install
```

### How to use

There are three types of augmentations which can be used:

- word2vec 

```python
from textaugment import Word2vec
```

- wordnet 
```python
from textaugment import Wordnet
```
- translate (This will require internet access)
```python
from textaugment import Translate
```
#### Word2vec-based augmentation

[See this notebook for an example](https://github.com/dsfsi/textaugment/blob/master/examples/word2vec_example.ipynb)

**Basic example**

```python
>>> from textaugment import Word2vec
>>> t = Word2vec(model='path/to/gensim/model'or 'gensim model itself')
>>> t.augment('The stories are good')
The films are good
```
**Advanced example**

```python
>>> runs = 1 # By default.
>>> v = False # verbose mode to replace all the words. If enabled runs is not effective. Used in this paper (https://www.cs.cmu.edu/~diyiy/docs/emnlp_wang_2015.pdf)
>>> p = 0.5 # The probability of success of an individual trial. (0.1<p<1.0), default is 0.5. Used by Geometric distribution to selects words from a sentence.

>>> t = Word2vec(model='path/to/gensim/model'or'gensim model itself', runs=5, v=False, p=0.5)
>>> t.augment('The stories are good')
The movies are excellent
```
#### WordNet-based augmentation
**Basic example**
```python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('wordnet')
>>> from textaugment import Wordnet
>>> t = Wordnet()
>>> t.augment('In the afternoon, John is going to town')
In the afternoon, John is walking to town
```
**Advanced example**

```python
>>> v = True # enable verbs augmentation. By default is True.
>>> n = False # enable nouns augmentation. By default is False.
>>> runs = 1 # number of times to augment a sentence. By default is 1.
>>> p = 0.5 # The probability of success of an individual trial. (0.1<p<1.0), default is 0.5. Used by Geometric distribution to selects words from a sentence.

>>> t = Wordnet(v=False ,n=True, p=0.5)
>>> t.augment('In the afternoon, John is going to town')
In the afternoon, Joseph is going to town.
```
#### RTT-based augmentation
**Example**
```python
>>> src = "en" # source language of the sentence
>>> to = "fr" # target language
>>> from textaugment import Translate
>>> t = Translate(src="en", to="fr")
>>> t.augment('In the afternoon, John is going to town')
In the afternoon John goes to town
```
# EDA: Easy data augmentation techniques for boosting performance on text classification tasks 
## This is the implementation of EDA by Jason Wei and Kai Zou. 

https://www.aclweb.org/anthology/D19-1670.pdf

[See this notebook for an example](https://github.com/dsfsi/textaugment/blob/master/examples/eda_example.ipynb)

#### Synonym Replacement
Randomly choose *n* words from the sentence that are not stop words. Replace each of these words with
one of its synonyms chosen at random. 

**Basic example**
```python
>>> from textaugment import EDA
>>> t = EDA()
>>> t.synonym_replacement("John is going to town")
John is give out to town
```

#### Random Deletion
Randomly remove each word in the sentence with probability *p*.

**Basic example**
```python
>>> from textaugment import EDA
>>> t = EDA()
>>> t.random_deletion("John is going to town", p=0.2)
is going to town
```

#### Random Swap
Randomly choose two words in the sentence and swap their positions. Do this n times.

**Basic example**
```python
>>> from textaugment import EDA
>>> t = EDA()
>>> t.random_swap("John is going to town")
John town going to is
```

#### Random Insertion 
Find a random synonym of a random word in the sentence that is not a stop word. Insert that synonym into a random position in the sentence. Do this n times

**Basic example**
```python
>>> from textaugment import EDA
>>> t = EDA()
>>> t.random_insertion("John is going to town")
John is going to make up town
```

# Mixup augmentation

This is the implementation of mixup augmentation by [Hongyi Zhang, Moustapha Cisse, Yann Dauphin, David Lopez-Paz](https://openreview.net/forum?id=r1Ddp1-Rb) adapted to NLP. 

Used in [Augmenting Data with Mixup for Sentence Classification: An Empirical Study](https://arxiv.org/abs/1905.08941). 

Mixup is a generic and straightforward data augmentation principle. In essence, mixup trains a neural network on convex combinations of pairs of examples and their labels. By doing so, mixup regularises the neural network to favour simple linear behaviour in-between training examples. 

## Implementation

[See this notebook for an example](https://github.com/dsfsi/textaugment/blob/master/examples/mixup_example_using_IMDB_sentiment.ipynb)

## Built with ❤ on
* [Python](http://python.org/)

## Authors
* [Joseph Sefara](https://za.linkedin.com/in/josephsefara) (http://www.speechtech.co.za)
* [Vukosi Marivate](http://www.vima.co.za) (http://www.vima.co.za)

## Acknowledgements
Cite this [paper](https://link.springer.com/chapter/10.1007%2F978-3-030-57321-8_21) when using this library. [Arxiv Version](https://arxiv.org/abs/1907.03752)

```
@inproceedings{marivate2020improving,
  title={Improving short text classification through global augmentation methods},
  author={Marivate, Vukosi and Sefara, Tshephisho},
  booktitle={International Cross-Domain Conference for Machine Learning and Knowledge Extraction},
  pages={385--399},
  year={2020},
  organization={Springer}
}
```

## Licence
MIT licensed. See the bundled [LICENCE](https://github.com/dsfsi/textaugment/blob/master/LICENCE) file for more details.
