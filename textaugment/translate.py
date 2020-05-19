#!/usr/bin/env python
# WordNet-based data augmentation 
#
# Copyright (C) 2020
# Author: Joseph Sefara
# URL: <https://github.com/dsfsi/textaugment/>
# For license information, see LICENSE

from .constants import LANGUAGES
from textblob import TextBlob
from textblob.translate import NotTranslated
from googletrans import Translator


class Translate: 
    """
    A set of functions used to augment data.
    Supported languages are:
    Language Name	Code
    Afrikaans	af
    Albanian	sq
    Arabic	ar
    Azerbaijani	az
    Basque	eu
    Bengali	bn
    Belarusian	be
    Bulgarian	bg
    Catalan	ca
    Chinese Simplified	zh-CN
    Chinese Traditional	zh-TW
    Croatian	hr
    Czech	cs
    Danish	da
    Dutch	nl
    English	en
    Esperanto	eo
    Estonian	et
    Filipino	tl
    Finnish	fi
    French	fr
    Galician	gl
    Georgian	ka
    German	de
    Greek	el
    Gujarati	gu
    Haitian Creole	ht
    Hebrew	iw
    Hindi	hi
    Hungarian	hu
    Icelandic	is
    Indonesian	id
    Irish	ga
    Italian	it
    Japanese	ja
    Kannada	kn
    Korean	ko
    Latin	la
    Latvian	lv
    Lithuanian	lt
    Macedonian	mk
    Malay	ms
    Maltese	mt
    Norwegian	no
    Persian	fa
    Polish	pl
    Portuguese	pt
    Romanian	ro
    Russian	ru
    Serbian	sr
    Slovak	sk
    Slovenian	sl
    Spanish	es
    Swahili	sw
    Swedish	sv
    Tamil	ta
    Telugu	te
    Thai	th
    Turkish	tr
    Ukrainian	uk
    Urdu	ur
    Vietnamese	vi
    Welsh	cy
    Yiddish	yi

    Example usage: ::
        >>> from textaugment import Translate
        >>> t = Translate(src="en",to="es")
        >>> t.augment('I love school')
        i adore school
    """

    def __init__(self, **kwargs):

        """
        A method to initialize parameters

        :type src: str
        :param src: Source language of the text
        :type to: str
        :param to: Destination language to translate to. The language should be a family of the source language for
                better results. The text will then be translated back to the source language.
        :rtype:   None
        :return:  Constructer do not return.
        """
        hl = LANGUAGES
        
        try:
            if "to" not in kwargs:
                raise ValueError("'to' missing")
            elif "src" not in kwargs:
                raise ValueError("'src' missing")
            if kwargs['to'] not in hl:
                raise KeyError("Value of to is not surpported. See help(Translate)")
            if kwargs['src'] not in hl:
                raise KeyError("Value of src is not surpported. See help(Translate)")
        except (ValueError, KeyError):
            print("The values of the keys 'to' and 'src' are required. E.g Translate(src='en', to='es')")
            raise
        else:    
            self.to = kwargs['to']
            self.src = kwargs['src']

    def augment(self, data):
        """
        A method to paraphrase a sentence.
        
        :type data: str
        :param data: sentence used for data augmentation 
        :rtype:   str
        :return:  The augmented data
        """
        if type(data) is not str:
            raise TypeError("DataType must be a string")
        data = TextBlob(data.lower())
        try:
            data = data.translate(from_lang=self.src, to=self.to)
            data = data.translate(from_lang=self.to, to=self.src)
        except NotTranslated:
            try:  # Switch to googletrans to do translation.
                translator = Translator()
                data = translator.translate(data, dest=self.to, src=self.src).text
                data = translator.translate(data, dest=self.src, src=self.to).text
            except Exception:
                print("Error Not translated.\n")
                raise

        return str(data).lower()
