# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:34:03 2019

@author: sachin.mandhotra
"""
# =============================================================================
# Using NLP for train  set
# =============================================================================
import pandas as pd
import numpy as np
from datetime import datetime

start_time = datetime.now()
print("start time is =",start_time)
# Importing the dataset
dataset = pd.read_csv('train.csv',  encoding = "ISO-8859-1")
dataset = dataset.fillna("Null")
dataset = dataset[["Title",'TRANS_CONV_TEXT', 'Patient_Tag']]

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
#stopwords is a megalist of all the non-useful words for our model(e.g that, at ,this will be removed)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#portstemmer is used to extract the root of the word .(e.g loved becomes love)
corpus = []
corpus1 = []
#Initializing a list that will hold all the words that can help our model
for i in range(0, 1157):
    review = re.sub('[^a-zA-Z]', ' ', str(dataset['TRANS_CONV_TEXT'][i]))
    #Fetching only the alphabets, and celaning everything else
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #the above line will remove words like that , and convert loved--> love
    review = ' '.join(review)
    #joining the concerned words
    corpus.append(review)
    #Making the list of all those words

for z in range(0,1157):
    #doing same for next column
    review1 = re.sub('[^a-zA-Z]', ' ', str(dataset['Title'][z]))
    #Fetching only the alphabets, and celaning everything else
    review1 = review1.lower()
    review1 = review1.split()
    ps = PorterStemmer()
    review1 = [ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]
    #the above line will remove words like that , and convert loved--> love
    review1 = ' '.join(review1)
    #joining the concerned words
    corpus1.append(review1)
    #Making the list of all those words

## Creating the Bag of Words model
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features = 1400)
#X1 = pd.DataFrame(cv.fit_transform(corpus).toarray())
#X2 = pd.DataFrame(cv.fit_transform(corpus1).toarray())


from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 1400)
X1 = pd.DataFrame(cv.fit_transform(corpus).toarray())
X2 = pd.DataFrame(cv.fit_transform(corpus1).toarray())


#MERGING to create Sparse matrix
X = pd.concat([X1,X2],axis=1)


y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn import metrics
#Accuracy of our model
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy is calculated as sum of first diagonal divided by total examples !
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-Score:",metrics.f1_score(y_test, y_pred))


# =============================================================================
# For Test Set
# =============================================================================


#Now, do the same for test data
data = pd.read_csv('test.csv',  encoding = "ISO-8859-1")
data = data.fillna("Null")
data = data[["Title",'TRANS_CONV_TEXT']]


# Cleaning the texts
import re
#import nltk
#nltk.download('stopwords')
#stopwords is a megalist of all the non-useful words for our model(e.g that, at ,this will be removed)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#portstemmer is used to extract the root of the word .(e.g loved becomes love)
corpus2 = []
corpus3 = []
#Initializing a list that will hold all the words that can help our model
for i in range(0, 571):
    review = re.sub('[^a-zA-Z]', ' ', str(data['TRANS_CONV_TEXT'][i]))
    #Fetching only the alphabets, and celaning everything else
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #the above line will remove words like that , and convert loved--> love
    review = ' '.join(review)
    #joining the concerend words
    corpus2.append(review)
    #Making the list of all those words
    
for z in range(0,571):
    #doing same for next column
    review1 = re.sub('[^a-zA-Z]', ' ', str(data['Title'][z]))
    #Fetching only the alphabets, and cleaning everything else
    review1 = review1.lower()
    review1 = review1.split()
    ps = PorterStemmer()
    review1 = [ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]
    #the above line will remove words like that , and convert loved--> love
    review1 = ' '.join(review1)
    #joining the concerned words
    corpus3.append(review1)
    #Making the list of all those words

    
## Creating the Bag of Words model
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features = 1400)
#X3 = pd.DataFrame(cv.fit_transform(corpus2).toarray())
#X4 = pd.DataFrame(cv.fit_transform(corpus3).toarray())

##using alternate method
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 1400)
X3 = pd.DataFrame(cv.fit_transform(corpus2).toarray())
X4 = pd.DataFrame(cv.fit_transform(corpus3).toarray())

#joining these two into one sparse matrix
FF = pd.concat([X3,X4],axis=1)


# Predicting the Test set results
y_pred_test = classifier.predict(FF)


## Converting data type of y_pred_test
index_vals = np.arange(1,572)
y_pred_test = pd.DataFrame(y_pred_test, index = index_vals)
y_pred_test = pd.DataFrame(y_pred_test).astype('int64')
y_pred_test.index.name = 'Index'
y_pred_test.to_csv('sachin_submission.csv',header=['Patient_Tag'])

end_time = datetime.now()
print("End time is=",end_time)

total_time = (end_time) - (start_time)
print("Total run time =",total_time)
