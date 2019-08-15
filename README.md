# [TextAugment: Improving Short Text Classification through Global Augmentation Methods](https://arxiv.org/abs/1907.03752) 

[![licence](https://img.shields.io/github/license/dsfsi/textaugment.svg?maxAge=3600)](https://github.com/dsfsi/textaugment/blob/master/LICENCE) [![GitHub release](https://img.shields.io/github/release/dsfsi/textaugment.svg?maxAge=3600)](https://github.com/dsfsi/textaugment/releases) [![Wheel](https://img.shields.io/pypi/wheel/textaugment.svg?maxAge=3600)](https://pypi.python.org/pypi/textaugment) [![python](https://img.shields.io/pypi/pyversions/textaugment.svg?maxAge=3600)](https://pypi.org/project/textaugment/)

## You have just found TextAugment.

TextAugment is a Python 3 library for augmenting text for natural language processing applications. TextAugment stands on the giant shoulders of [NLTK](https://www.nltk.org/), [Gensim](https://radimrehurek.com/gensim/), and [TextBlob](https://textblob.readthedocs.io/) and plays nicely with them.

## Citation Paper

**[Improving short text classification through global augmentation methods](https://arxiv.org/abs/1907.03752)**.

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
## Built with ❤ on
* [Python](http://python.org/)

## Authors
* [Joseph Sefara](https://za.linkedin.com/in/josephsefara) (http://www.speechtech.co.za)
* [Vukosi Marivate](http://www.vima.co.za) (http://www.vima.co.za)

## Acknowledgements
Cite this [paper](https://arxiv.org/abs/1907.03752) when using this library.

```
@article{marivate2019improving,
  title={Improving short text classification through global augmentation methods},
  author={Marivate, Vukosi and Sefara, Tshephisho},
  journal={arXiv preprint arXiv:1907.03752},
  year={2019}
}
```

## Licence
MIT licensed. See the bundled [LICENCE](https://github.com/dsfsi/textaugment/blob/master/LICENCE) file for more details.
