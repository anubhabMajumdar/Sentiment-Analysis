import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk
import nltk.data
import logging
from gensim.models import word2vec
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from random import shuffle

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

def writeSentences(reviews, f):
    count = 0
    for review in reviews:
        sentences = review_to_sentences(review.decode("utf8"), tokenizer, True)
        result = ""
        for s in sentences:
            result += " ".join(s)
        result += "\n"    
        f.write(result)  
        count += 1
        print count, " done"  



# ****************************************************************

# nltk.download()   
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# Read data from files 
train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

# Verify the number of reviews that were read (100,000 in total)
print "Read %d labeled train reviews, %d labeled test reviews, " \
"and %d unlabeled reviews\n" % (train["review"].size,  
 test["review"].size, unlabeled_train["review"].size )


sentences = []  # Initialize an empty list of sentences

# write data
f_test = open("test.txt", "a")
f_trainLabeled_pos = open("train-pos.txt", "a")
f_trainLabeled_neg = open("train-neg.txt", "a")
f_trainUnLabeled = open("train-unsup.txt", "a")

print "Writing test data"
writeSentences(test['review'], f_test)
f_test.close() 

print "Writing unlabeled train data"
writeSentences(unlabeled_train['review'], f_trainUnLabeled)
f_trainUnLabeled.close()

print "Writing positive train data"
writeSentences((train.loc[train['sentiment'] == 1])['review'], f_trainLabeled_pos)
f_trainLabeled_pos.close()

print "Writing negative train data"
writeSentences((train.loc[train['sentiment'] == 0])['review'], f_trainLabeled_neg)
f_trainLabeled_neg.close()
