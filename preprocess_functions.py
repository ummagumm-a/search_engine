import pandas as pd

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


import string
import re
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk import word_tokenize

sw_eng = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def text_preprocess(text):
    text = str(text)
   
    # eliminate punctuation
    p = re.compile("[" + re.escape(string.punctuation + '®—‘’“”»…') + "]")
    text = p.sub("", text.lower())

    text = re.sub("\n", " ", text)
    text = re.sub("http\w+| \d+", "", text)

    # eliminate stopwords
    # text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if not word in sw_eng])
    text = ' '.join([word for word in word_tokenize(text) if not word in sw_eng])
    
    # lemmatize
    pos_tagged = pos_tag(word_tokenize(text))
    pos_tagged = [(word, get_wordnet_pos(tag)) for word, tag in pos_tagged]

    text = ' '.join([lemmatizer.lemmatize(word, tag)
                            for word, tag in pos_tagged])

    

    return text
