from flask import Flask, render_template, request
#https://www.kaggle.com/code/karakoza22/sentiment-analysis
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize,regexp_tokenize
import seaborn as sns
import warnings
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
warnings.filterwarnings('ignore')
# mpl.style.use("ggplot")
from textblob import TextBlob
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
from nltk.corpus import stopwords
from gensim.models.tfidfmodel import TfidfModel
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from langdetect import detect_langs
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import contractions
from tqdm import tqdm
# import plotly.express as px
df = pd.read_csv("F:/SE project/IMDB Dataset.csv")

def cleaning_sentiment(df,col_name):
    porter = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    w = df[col_name].apply(lambda x :x.lower())
    w = w.apply(lambda x: re.sub('(<.*?>)|(\n)|([^\w\s\.\,])|([_])|([.])|([,])|(\s\s+)|([ا-ي])','',x))
    w = w.apply(lambda x: contractions.fix(x))
    w = w.apply(lambda x: ' '.join([porter.stem(i) for i in word_tokenize(x) if i not in stop_words]))
    return w
def find_duplicates(arr):
    d = {}
    duplicates = []
    for s in arr:
        if s in d:
            d[s] += 1
        else:
            d[s] = 1
    for s in d:
        if d[s] > 1:
            duplicates.append(s)
    return duplicates
def extracting_len_of_sentiment (df,col_name):
   w = df[col_name].apply(lambda x :x.lower())
   len_of_review = [len(word_tokenize(i)) for i in w] 
   return len_of_review

def predict_sentiment(review, vectorizer, model):
    review_tfidf_vectorizer = vectorizer.transform([review])
    sentiment = model.predict(review_tfidf_vectorizer)[0]
    print (sentiment)
    if sentiment == 0:
        return 'Negative'
    elif sentiment == 1:
        return 'Positive'
    else:
        return 'Unknown'

        
df['review'] = cleaning_sentiment(df,'review')
# df.to_csv('clean_data.csv')
df=pd.read_csv('clean_data.csv')
df['n_of_words'] = extracting_len_of_sentiment(df,'review')
df['list_of_words'] = df['review'].apply(lambda x:str(x).split()) 
df.replace(to_replace="positive",value=1,inplace=True)
df.replace(to_replace="negative",value=0,inplace=True)
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
tf =TfidfVectorizer()
X_train_tfidf_vectorizer=tf.fit_transform(X_train)
X_test_tfidf_vectorizer=tf.transform(X_test)
print(X_train_tfidf_vectorizer.shape, X_test_tfidf_vectorizer.shape)



ones = [i for i in y_train if i==1]
print(f"model trained on: {len(ones)} of ones, while: {(len(y_train)-len(ones))} of zeros")
X_train=X_train_tfidf_vectorizer
X_test=X_test_tfidf_vectorizer
#models = [("logistic Regression",LogisticRegression()),("Decision Tree",DecisionTreeClassifier()), ("MultinomialNB",MultinomialNB())]
model=LogisticRegression()
clf_1 = model.fit(X_train,y_train)
y_pred=clf_1.predict(X_test)

print(f"Recall score: {round(recall_score(y_test,y_pred),3)}, precision score: {round(precision_score(y_test,y_pred),3)}, f1-score: {round(f1_score(y_test,y_pred),3)},accuracy score: {round(accuracy_score(y_test,y_pred),3)}\n")
print(pd.DataFrame(confusion_matrix(y_test,y_pred)))    
print(f"number of unique predicted classes: {np.unique(y_pred)}")

model=DecisionTreeClassifier()
clf_2 = model.fit(X_train,y_train)
y_pred=clf_2.predict(X_test)
print(f"Recall score: {round(recall_score(y_test,y_pred),3)}, precision score: {round(precision_score(y_test,y_pred),3)}, f1-score: {round(f1_score(y_test,y_pred),3)},accuracy score: {round(accuracy_score(y_test,y_pred),3)}\n")
print(pd.DataFrame(confusion_matrix(y_test,y_pred)))    
print(f"number of unique predicted classes: {np.unique(y_pred)}")


model=MultinomialNB()
clf_3 = model.fit(X_train,y_train)
y_pred=clf_3.predict(X_test)
print(f"Recall score: {round(recall_score(y_test,y_pred),3)}, precision score: {round(precision_score(y_test,y_pred),3)}, f1-score: {round(f1_score(y_test,y_pred),3)},accuracy score: {round(accuracy_score(y_test,y_pred),3)}\n")
print(pd.DataFrame(confusion_matrix(y_test,y_pred)))    
print(f"number of unique predicted classes: {np.unique(y_pred)}")



# assume that 'tf' and 'clf' are the vectorizer and model objects, respectively
new_review = "i hate this movie"
predicted_sentiment = predict_sentiment(new_review, tf, clf_1)
print(predicted_sentiment)
predicted_sentiment = predict_sentiment(new_review, tf, clf_2)
print(predicted_sentiment)
predicted_sentiment = predict_sentiment(new_review, tf, clf_3)
print(predicted_sentiment)



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        inp1 = request.form['input1']
        inp2 = request.form['input2']
        temp=[]
        display(inp1, inp2)
        temp.append(predict_sentiment(inp2, tf, clf_1))
        # print(predicted_sentiment)
        temp.append(predict_sentiment(inp2, tf, clf_2))
        # print(predicted_sentiment)
        temp.append(predict_sentiment(inp2, tf, clf_3))
        # print(predicted_sentiment)
        
        inp3 = 'the output will b here' # data processing and function here ___________________
        
        out1 = find_duplicates(temp)
        return render_template('process.html', inp1=inp1, inp2=inp2 , inp3=out1)
    else:
        return render_template('index.html')


#you can add ur ml model here or send the files to a different function and do the processing there
def display(inp1, inp2):
    print(inp1, inp2)

if __name__=='__main__':
    app.run()