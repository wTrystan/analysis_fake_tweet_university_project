import pandas as pd
import sklearn
import numpy as np
from langdetect import detect
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

pattern = r'''(?x)          # set flag to allow verbose regexps
         (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
       | \w+(?:-\w+)*        # words with optional internal hyphens
       | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
       | \.\.\.              # ellipsis
       | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
     '''

train = pd.read_csv('C:\\Users\\tryst\\Desktop\\Master 2 SIAD\\pythonProject\\fake-news\\train.csv')

df = pd.DataFrame(train)

print(df.columns)

print(train.isnull().sum())

freq = train['label'].value_counts()
print(freq)

trainRN = train[train['label'] == 0]
trainFN = train[train['label'] == 1]

baseRN = []
baseFN = []


method = trainRN['title'][4323:4350]
print(str(method))

for l in trainFN['text']:
    tokenized_word = nltk.regexp_tokenize(str(l), pattern)
    for w in tokenized_word:
        baseFN.append(w)

for l in trainRN['text']:
    tokenized_word = nltk.regexp_tokenize(str(l), pattern)
    for w in tokenized_word:
        baseRN.append(w)


baseRN = [word for word in baseRN if word]
baseFN = [word for word in baseFN if word]

from collections import Counter
print(len(baseRN))
baseFN = list(set(baseFN))
print(len(baseRN))


texta = ["Classification des tweets", "Tweets transformés", "Modération du site Twitter"]

vectorizer = TfidfVectorizer()
vectorizer.fit(texta)

textavector = vectorizer.transform(texta)

print(textavector)


groupby = []

groupbysum = df.groupby(['author']).sum()
groupbycount = df.groupby(['author']).count()



########
