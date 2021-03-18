# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 19:04:48 2021

@author: Sai Krishna
"""

#Import libraries (almost only need to install textblob, tweepy)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tweepy
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Authentication
#Get Twitter developer account. Create a project. API key is consumer key. Generate access token and secret as well.
#These keys can be regenerated and used as well. 
#Keys are anonymized for privacy reasons.

consumerKey = 'aaaaaaaaaaa'
consumerSecret = 'bbbbbbbbbbbbbbbbbbbbbbbb'
accessToken = 'ccccccccccccccccccccccccccccccccccccccccccccccccccccc'
accessTokenSecret = 'dddddddddddddddddddddddddddddddddddddddddddd'

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

#Sentiment Analysis function. Give the name and # of tweets to analyze in the function.
def percentage(part,whole):
 return 100 * float(part)/float(whole)
keyword = '#YuvrajSingh'
noOfTweet = 100
tweets = tweepy.Cursor(api.search, q=keyword).items(noOfTweet)
positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []

for tweet in tweets:
 
 #print(tweet.text)
 tweet_list.append(tweet.text)
 analysis = TextBlob(tweet.text)
 score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
 
 neg = score['neg']
 neu = score['neu']
 pos = score['pos']
 comp = score['compound']
 polarity += analysis.sentiment.polarity
 
 if neg > pos:
     negative_list.append(tweet.text)
     negative += 1
 elif pos > neg:
     positive_list.append(tweet.text)
     positive += 1
 
 elif pos == neg:
     neutral_list.append(tweet.text)
     neutral += 1

#Using the created function
positive = percentage(positive, noOfTweet)
negative = percentage(negative, noOfTweet)
neutral = percentage(neutral, noOfTweet)
polarity = percentage(polarity, noOfTweet)
positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')

tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)

print('total number: ',len(tweet_list))
print('positive number: ',len(positive_list))
print('negative number: ', len(negative_list))
print('neutral number: ',len(neutral_list))

tweet_list.head()
negative_list.head()
#Though some tweets seem to be positive, could be based on the backend algorithm.
positive_list.head()