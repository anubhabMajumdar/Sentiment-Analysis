import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm

max_features = 3000

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "html.parser").get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   



######################################################################################

def get_bow(df):

	clean_data = []

	print "Cleaning data"

	for i in df['review']:
		clean_data.append(review_to_words(i))

	######################################################################################

	print "Creating bag of word model"

	
	vectorizer = CountVectorizer(analyzer = "word",   \
	                             tokenizer = None,    \
	                             preprocessor = None, \
	                             stop_words = None,   \
	                             max_features = max_features) 

	train_data_features = vectorizer.fit_transform(clean_data)

	train_data_features = train_data_features.toarray()

	return train_data_features
	# return df['sentiment'].tolist()

######################################################################################

print "Reading data"

df = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)


######################################################################################

train_data_features = get_bow(df)

print "Training classifier"

#################################### Random Forest ################################################

# forest = RandomForestClassifier(n_estimators = 100) 
# classifier = forest.fit( train_data_features, df['sentiment'] )

####################################### KNN ###########################################

# neigh = KNeighborsClassifier(n_neighbors=3)
# classifier = neigh.fit(train_data_features, df['sentiment'])


######################################################################################

# gnb = GaussianNB()
# classifier = gnb.fit(train_data_features, df['sentiment'])


######################################################################################

# clf = svm.LinearSVC()
clf = svm.SVC()
classifier = clf.fit(train_data_features, df['sentiment'])


######################################################################################

# clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(max_features, ), random_state=1)
# classifier = clf.fit(train_data_features, df['sentiment'])


######################################################################################

print "Reading data"

df = pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3)


######################################################################################

test_data_features = get_bow(df)

print "Testing classifier"

# Use the random forest to make sentiment label predictions
result = classifier.predict(test_data_features)

######################################################################################

print "Writing predictions"

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":df["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )






