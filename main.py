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

# nltk.download('punkt')

train2 = pd.read_csv('C:\\Users\\tryst\\Desktop\\Master 2 SIAD\\pythonProject\\fake-news\\submit.csv')
train1 = pd.read_csv('C:\\Users\\tryst\\Desktop\\Master 2 SIAD\\pythonProject\\fake-news\\test.csv')
#train = pd.read_csv('C:\\Users\\tryst\\Desktop\\Master 2 SIAD\\pythonProject\\fake-news\\train.csv')

train = pd.merge(train1,train2, on=["id"])
print(train)

## On remplace valeur vide de Text par NaN
train['text'] = train['text'].replace('', np.nan)
train.dropna(subset=['text'], inplace=True)
train['text'] = train['text'].str.lower()
text = train['text']

#train.drop( (detect(train['text']) != 'en').index, inplace=True)
# nltk.download('stopwords')

textcorrige = []
sentences = []
tableFinale = []

texteSimple = text[0:50]

pattern = r'''(?x)          # set flag to allow verbose regexps
        (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | \.\.\.              # ellipsis
      | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
    '''

## On va tokeniser et retirer les stop words
def transformation(table) :
    filtered_sent = []
    tokenized_word=nltk.regexp_tokenize(table, pattern)
    stop_words=set(stopwords.words("english"))
    for w in tokenized_word:
            filtered_sent.append(w)
    filtered_sent = ' '.join(filtered_sent)
    return filtered_sent

for i in range(0,50):
    tableFinale.append(transformation(texteSimple[i]))

print(tableFinale)

#DataFrame de nos trucs
tableMerge = {'tweets':tableFinale,'label':train['label'][0:50]}
tableMerge = pd.DataFrame(tableMerge)

# Vectorization de la table

vectorizer = TfidfVectorizer()
vectorizer.fit(tableMerge['tweets'])

vectorTrainTweet = vectorizer.transform(tableMerge['tweets'])
TrainLabel = tableMerge['label']


# Regression logistique

model = LogisticRegression()
model.fit(vectorTrainTweet, TrainLabel)

# On teste la précision de notre modèle :

predictionTweetTrain = model.predict(vectorTrainTweet)
precisionTrain = accuracy_score(predictionTweetTrain, TrainLabel)

print('Le score du modèle sur les données d entrainement est', precisionTrain)


# On teste la précision de notre modèle sur nos données test :

predictionTweetTest = model.predict(vectorTestTweet)
precisionTest = accuracy_score(predictionTweetTest, vectorTestLabel)

print('Le score du modèle sur les données test est', precisionTest)


#Donc là, on va lancer le code en rajoutant à chaque fois des améliorations, et on va lancer ce bout de code restant quand on aura la meilleure version :
