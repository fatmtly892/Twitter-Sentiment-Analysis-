#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This is a Natural Language Processing Project to analyze the sentiment of Twitter users.The objective of this project is to detect hate speech in tweets.
# 
# #Project Objectives:
# Apply python libraries to import and visualize datasets
# Perform exploratory data analysis and plot word-cloud
# Perform text data cleaning such as removing punctuation and stop words
# Understand the concept of count vectorization (tokenization)
# Perform tokenization to tweet text using Scikit Learn
# Understand the theory and intuition behind Naïve Bayes classifiers
# Understand the difference between prior probability, posterior probability and likelihood
# Train Naïve Bayes classifier models using Scikit-Learn to preform classification
# Evaluate the performance of trained Naïve Bayes Classifier model using confusion matrices
# 

# Task #1: Understand the Problem Statement and business case:
# Natural Language Processors (NLP) work by converting words(text) to numbers.
# These numbers are then used to train ML/Ai models to make predictions.
# Predictions could be sentiment inferred from social media posts and product reviews.
# AI/Ml-based sentiment analysis is crucial for companies to predict whether their customers happy or not.
# The process could be done automatically without having humans manually review thousand of tweets or customers reviews.
# In this case study, we will analyze thousands of Twitter tweets to predict peoples sentiment.
# ![0_HfnMUXqFhcY966kq.png](attachment:0_HfnMUXqFhcY966kq.png).

# In[ ]:


#Task #2: Import libraries and datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


# In[6]:


tweet_df=pd.read_csv('tweeter data')


# In[3]:


tweet_df.head()


# In[4]:


tweet_df.info()


# In[5]:


tweet_df.describe()


# In[ ]:


tweet_df[['tweet']]


# In[8]:


#droping the'id'  column from the dataframe
tweet_df=tweet_df.drop('id', axis= 1)
tweet_df


# In[9]:


#Task #3: Perform Exploratory Data Analysis
sns.heatmap(tweet_df.isnull(), yticklabels= False, cbar= False, cmap='Blues')


# In[10]:


#Showing the distribution
sns.countplot(tweet_df['label'], label= 'count')


# In[11]:


# adding a length of tweets 
tweet_df['length'] = tweet_df['tweet'].apply(len)
tweet_df
                                            


# In[12]:


#showing the distribution of length
tweet_df['length'].plot(bins=100, kind ='hist')


# In[13]:


#explore the new dataset
tweet_df.describe()


# In[14]:


#exploring the shortest tweets
tweet_df[tweet_df['length']== 11]['tweet'].iloc[0]


# In[15]:


tweet_df[tweet_df['length']== 84]['tweet'].iloc[0]


# In[16]:


#divide the dataset into positive & negative tweets

positive= tweet_df[tweet_df['label']== 0]
positive


# In[17]:


negative = tweet_df[tweet_df['label']== 1]
negative


# In[18]:


#Task #4: Plot the word cloud
#converting the tweet column into a list
sentences = tweet_df['tweet'].tolist()


# In[19]:


#converting sentences list into one string
sentences_as_one_string = " ".join(sentences)


# In[20]:


sentences_as_one_string


# In[21]:


#install wordcloud module
#ploting a wordcloud for all tweets
get_ipython().system('pip install Wordcloud')
from wordcloud import WordCloud
plt.figure(figsize=(20, 20))
plt.imshow(WordCloud().generate(sentences_as_one_string))
plt.show()


# In[22]:


#ploting a wordcloud for negative tweets
negative_tweets = negative['tweet'].tolist()
negative_as_one_string = " ".join(negative_tweets)
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(negative_as_one_string))


# In[ ]:


#: Create a pipeline to remove stop-words, punctuation, and perform tokenization and vectorization 
def tweets_cleaning(tweet):
   tweets_without_punctuation = [char for char in tweet if char not in string.punctuation]
   tweet_join = ''.join(tweets_without_punctuation)
   tweets_without_stopwords = [word for word in tweet_join if word.lower() not in stopwords.words('english')]
   return tweets_without_stopwords
#converting the clean tweets into a Bag of Words
tweets_vectorizer = CountVectorizer(analyzer = tweets_cleaning, dtype = 'uint8').fit_transform(tweet_df['tweet']).toarray()


# In[ ]:


tweets_vectorizer.shape


# In[ ]:


tweets_clean = tweet_df['tweet'].apply(tweets_cleaning)#lets test the new added function
print(tweets_clean[5])


# In[ ]:


#it is the time of train a machine learning model
X = tweets_vectorizer

y = tweet_df['label']


# In[ ]:


#Task #9: Understand the theory and intuition behind Naive Bayes classifiers
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, stratify = y)

#instaniate the classifier 
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train) #fitting the model

#predicting the labels of the test set
y_predict = NB_classifier.predict(X_test)


# In[ ]:


#Assess trained model performance through confusion_matrix
confusion = confusion_matrix(y_test, y_predict)

#ploting the confusion matrix
sns.heatmap(confusion, annot = True)


# In[ ]:


#getting the classification report 
print(classification_report(y_test, y_predict))

