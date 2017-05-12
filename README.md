# Abstract
Human beings generally communicate information, facts and sentiments through the spoken or written word. We do this very efficiently and from an early age. Humans are also very good at processing these communications - that is, by reading a document or hearing a speech , we can parse information, facts and sentiments of the author very quickly, efficiently and accurately. However, this is not an easy task for a machine to do. Researchers have been trying to solve some of these problems for a long time and they have achieved some remarkable success over the years. This particular branch of research is referred to as Natural Language Processing.
In this project, I focused on one particular task of natural language processing - sentiment analysis of written text. The project explores three approaches to represent written word as fixed length feature vectors and use classification algorithms to predict sentiments expressed by them. Using a standard dataset of movie reviews, the three approaches are compared through experiments and their limitations are noted to help identify future work.

# Dataset
+ IMDB movie review dataset [link]

# Feature Representation Algorithms
+ Bag-of-word model (baseline)
+ Distributed representation of words (Word2vec)
+ Distributed representation of paragraph (Doc2vec)

# Classification Algorithms
+ K Nearest Neighbour
+ Random Forest
+ Neural Network
+ Support Vector Machine

# Language and Libraries
* python 2.7
* Tensorflow
* numpy
* matplotlib
* pickle
* scipy
* gensim

# Instructions
* After installing the necessary libraries, run 
``
python bow.py
``
for bag-of-word model.
* Run 
``
python word2vec.py
``
for Word2vec model.
* Run 
``
python doc2vec.py
``
for Doc2vec model.

# Additional Resources
+ For introduction to text representation techniques - [projectPaper] 
+ Presentation - [presentation]

# References
* All the references are mentioned in [projectPaper].

   [projectPaper]: <https://drive.google.com/file/d/0BygLf1QZV3ixU3pKNjczT2RUa3c/view?usp=sharing>
   [presentation]: <https://docs.google.com/presentation/d/132b8VxEwQgSY-FSZeZ0u802cCjnbaMAcX8__jpPycRk/edit?usp=sharing4>
   [link]: <https://www.kaggle.com/c/word2vec-nlp-tutorial/data>
