# -*- coding: utf-8 -*-
"""
@author: sefaratj@gmail.com

setup.py must be in the root of the project
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
      name='textaugment',
      version='1.0',
      packages=setuptools.find_packages(),
      #  scripts= ['bin/test_video_pkg.py'],
      author='Joseph Sefara',
      author_email='sefaratj@gmail.com',
      license='MIT',
      keywords=['data augmentation', 'python', 'natural language processing'],
      url='https://pypi.org/project/textaugment',
      description='A library for augmenting text for natural language processing applications.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      #  Dependences
      #  install_requires= ['nltk', 'gensim','textblob','numpy','itertools','re'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ]
)
