import tensorflow as tf
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import random

max_features = 5000
batch_size = 100


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


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

######################################################################################


print "Reading data"

df = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
labels = df['sentiment'].tolist()
labels_matrix = []
for i in labels:
	if i == 0:
		labels_matrix.append([1,0])
	else:
		labels_matrix.append([0,1])		

######################################################################################

train_data_features = get_bow(df)

#################################### Random Forest ################################################

print "Reading data"

df = pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3)


######################################################################################

test_data_features = get_bow(df)

######################################################################################

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, max_features])
y = tf.placeholder(tf.float32, shape=[None, 2])

######################################################################################

w1 = weight_variable([max_features, 2*max_features])
b1 = bias_variable([2*max_features])

h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

######################################################################################

w2 = weight_variable([2*max_features, 2])
b2 = bias_variable([2])

y_nn = tf.matmul(h1, w2) + b2

######################################################################################

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_nn))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_nn,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

######################################################################################

print "Training"

for j in range(3000):

	random_index = random.sample(range(0, 25000), batch_size)

	batch_x = [train_data_features[i] for i in random_index]
	batch_y = [labels_matrix[i] for i in random_index]

	train_step.run(feed_dict={x: batch_x, y: batch_y})

	if j%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch_x, y: batch_y})
		print("step %d, training accuracy %g"%(j, train_accuracy))
  

	# break

######################################################################################

print "Testing"

predicted_class = tf.argmax(y_nn,1)
results = []

for i in xrange(0,len(test_data_features), batch_size):
	pl = predicted_class.eval(feed_dict={x:test_data_features[i:i+batch_size]})
	results.extend(pl)
	print i, " done"

######################################################################################

print "Writing predictions"

output = pd.DataFrame( data={"id":df["id"], "sentiment":results})

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

















