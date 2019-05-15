# [TextAugment: Improving short text classification through global augmentation methods]() 

TextAugment is a Python 3 library for augmenting text for natural language processing applications. TextAugment stands on the giant shoulders of [NLTK](https://www.nltk.org/), [Gensim](https://radimrehurek.com/gensim/), and [TextBlob](https://textblob.readthedocs.io/) and plays nicely with them.

## Citation Paper

**[Improving short text classification through global augmentation methods]()** published to MLDM

## Getting Started
This library uses input string data then output the augmented string data.  
There are three types of augmentations which can be activated as follows:

* word2vec = True / False

* wordnet = True / False

* translate = True / False (This will require internet access)

### Requirements

* Python 3

The following software packages are dependencies and will be installed automatically.

```shell
$ pip install nltk
$ pip install gensim
$ pip install textblob
$ pip install googletrans
$ pip install tools
$ pip install numpy
$ pip install itertools
$ pip install re
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
Use gensim to load a pre-trained word2vec model
The model can be downloaded from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)

Or training one from scratch using your data or the following datasets:

-[Text8 Wiki](http://mattmahoney.net/dc/enwik9.zip)

-[Dataset from "One Billion Word Language Modeling Benchmark"](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz)

### Installing

Install from pip
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

### How to import the library
```python
import textaugment
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Built With

* [Python](http://python.org/)

## Authors

* [Joseph Sefara](https://za.linkedin.com/in/josephsefara) 
* [Vukosi Marivate](http://www.vima.co.za) 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Cite this [paper](#) when using this library.

## Licence
MIT licensed. See the bundled [LICENCE](LICENCE) file for more details.
