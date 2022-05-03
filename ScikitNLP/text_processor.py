from typing import List
import re
import string
import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
stemmer = PorterStemmer()
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

# TODO: separate the regex patterns processor into separate functions
def process_tweet(tweet: str) -> str:
    '''
    Input: 
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    
    '''
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)

    return tweet

# TODO: this function could be generalized, as pass in a tokenizer, lemmatizer, and stopwords
def tokenizer_tweet(tweet, tokenizer, stopwords) -> List[str]:
    # tokenize tweets
    tweet_tokens = tokenizer.tokenize(tweet)

    tweet_clean_tokens = []
    for word in tweet_tokens:
        if (word not in stopwords and # remove stopwords
            word not in string.punctuation): # remove punctuation
            #tweets_clean.append(word)
            stem_word = stemmer.stem(word) # stemming word
            tweet_clean_tokens.append(stem_word)
    return tweet_clean_tokens
