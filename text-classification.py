import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#for text pre-processing
import re, string
import nltk
nltk.download('punkt',quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#for word embedding
import gensim
from gensim.models import Word2Vec


# convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text


# STOPWORD REMOVAL
def stopword(string):
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

#This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Tokenize the sentence
# LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string))  # Get position tags
    # Map the position tag and lemmatize the word/token
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)]
    return " ".join(a)

def finalpreprocess(string):
    #print(string)
    string = preprocess(string)
    #print(string)
    string = stopword(string)
    #print(string)
    string = lemmatizer(string)
    #print(string)
    return string

#for converting sentence to vectors/numbers from word vectors result by Word2Vec
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


df_train= pd.read_csv('disaster-tweets/train.csv')
df_test=pd.read_csv('disaster-tweets/test.csv')

# CLASS DISTRIBUTION
#if dataset is balanced or not
x=df_train['target'].value_counts()
#print(x)
#sns.barplot(x=x.index,y=x)
#plt.show()

#Missing values
#print(df_train.isna().sum())

#1. WORD-COUNT
df_train['word_count'] = df_train['text'].apply(lambda x: len(str(x).split()))
#Disaster tweets
print("Disaster Tweet word count:", df_train[df_train['target']==1]['word_count'].mean())
#Non-Disaster tweets
print("Non-Disaster Tweet word count:", df_train[df_train['target']==0]['word_count'].mean())
#Disaster tweets are more wordy than the non-disaster tweets

#2. CHARACTER-COUNT
df_train['char_count'] = df_train['text'].apply(lambda x: len(str(x)))
print("Disaster Tweet character count:", df_train[df_train['target']==1]['char_count'].mean()) #Disaster tweets
print("Non-Disaster Tweet character count:", df_train[df_train['target']==0]['char_count'].mean()) #Non-Disaster tweets
#Disaster tweets are longer than the non-disaster tweets

#3. UNIQUE WORD-COUNT
df_train['unique_word_count'] = df_train['text'].apply(lambda x: len(set(str(x).split())))
print("Disaster Tweet unique word count:", df_train[df_train['target']==1]['unique_word_count'].mean()) #Disaster tweets
print("Non-Disaster Tweet unique count:", df_train[df_train['target']==0]['unique_word_count'].mean()) #Non-Disaster tweets

#Plotting word-count per tweet
# fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
# train_words=df_train[df_train['target']==1]['word_count']
# ax1.hist(train_words,color='red')
# ax1.set_title('Disaster tweets')
# train_words=df_train[df_train['target']==0]['word_count']
# ax2.hist(train_words,color='green')
# ax2.set_title('Non-disaster tweets')
# fig.suptitle('Words per tweet')
# plt.show()

#cleaning text
df_train['clean_text'] = df_train['text'].apply(lambda x: finalpreprocess(x))
df_train = df_train.drop(columns=['word_count','char_count','unique_word_count'])

#create Word2vec vector
df_train['clean_text_tok'] = [nltk.word_tokenize(i) for i in df_train['clean_text']] #convert preprocessed sentence to tokenized sentence
print(df_train.head())
#min_count=1 means word should be present at least across all documents,
#if min_count=2 means if the word is present less than 2 times across all the documents then we shouldn't consider it
model = Word2Vec(df_train['clean_text_tok'],min_count=1)
w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))  #combination of word and its vector

# SPLITTING THE TRAINING DATASET INTO TRAINING AND VALIDATION

X_train, X_val, y_train, y_val = train_test_split(df_train["clean_text"],df_train["target"],test_size=0.2,shuffle=True)
X_train_tok = [nltk.word_tokenize(i) for i in X_train]  # for word2vec
X_val_tok = [nltk.word_tokenize(i) for i in X_val]  # for word2vec
print(X_val_tok)
# Word2vec
# Fit and transform
modelw = MeanEmbeddingVectorizer(w2v)
X_train_vectors_w2v = modelw.transform(X_train_tok)
X_val_vectors_w2v = modelw.transform(X_val_tok)

# FITTING THE CLASSIFICATION MODEL using Logistic Regression (W2v)
lr_w2v = LogisticRegression(solver='liblinear', C=10, penalty='l2')
lr_w2v.fit(X_train_vectors_w2v, y_train)  # model

# Predict y value for test dataset
y_predict = lr_w2v.predict(X_val_vectors_w2v)
y_prob = lr_w2v.predict_proba(X_val_vectors_w2v)[:, 1]

print(classification_report(y_val, y_predict))
print('Confusion Matrix:', confusion_matrix(y_val, y_predict))

fpr, tpr, thresholds = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)

#TF-IDF
# Convert x_train to vector since model can only run on numbers and not words- Fit and transform
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
#tfidf runs on non-tokenized sentences unlike word2vec
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train)
# Only transform x_test (not fit and transform)
X_val_vectors_tfidf = tfidf_vectorizer.transform(X_val)

#FITTING THE CLASSIFICATION MODEL using Logistic Regression(tf-idf)
lr_tfidf = LogisticRegression(solver='liblinear', C=10, penalty='l2')
lr_tfidf.fit(X_train_vectors_tfidf, y_train)  # model

# Predict y value for test dataset
y_predict = lr_tfidf.predict(X_val_vectors_tfidf)
y_prob = lr_tfidf.predict_proba(X_val_vectors_tfidf)[:, 1]

print(classification_report(y_val, y_predict))
print('Confusion Matrix:', confusion_matrix(y_val, y_predict))

fpr, tpr, thresholds = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)