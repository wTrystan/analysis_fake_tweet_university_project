import pandas as pd
import sklearn
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize, TweetTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from langdetect import detect


print(bloc)