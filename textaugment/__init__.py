import os
from .translate import Translate
from .word2vec import Word2vec
from .word2vec import Fasttext
from .wordnet import Wordnet
from .eda import EDA
from .aeda import AEDA
from .mixup import MIXUP
from .constants import LANGUAGES

name = "textaugment"

__version__ = '2.0.0'
__licence__ = 'MIT'
__author__ = 'Joseph Sefara'
__url__ = 'https://github.com/dsfsi/textaugment/'

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = [
    'Translate',
    'Word2vec',
    'Wordnet',
    'EDA',
    'AEDA',
    'MIXUP',
    'LANGUAGES'
]
