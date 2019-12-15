import gensim 
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer

#part 1
# file is downloaded, but because my computer sucks in dealing with it I'll do it using another file :)
# there are 30 sentences
# first 10 about cricket, next 10 about AI and last 10 about chemistry

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
 
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y

#reading data for both tasks

file = open("tmp2.txt", "r") 

sentences = []
words = []

for line in file:
	line = line.strip()
	cleaned = clean(line)
	words.append(cleaned)
	cleaned = ' '.join(cleaned)
	sentences.append(cleaned)

#print (data)
#print (sentences)

#part 2
# CBOW model is trained

model1 = gensim.models.Word2Vec(words, min_count = 1, size = 100, window = 5) 

#print("chemistry and ball : ", model1.wv.similarity('chemistry', 'ball')) 

#part 3
# TODO:

#part 4
# Clustering using KNN

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(sentences)

y_train = np.zeros(30)
y_train[10:20] = 1
y_train[20:30] = 2

modelknn = KNeighborsClassifier(n_neighbors=5)
modelknn.fit(X,y_train)

#part 5

def similar_words(word):
	return model1.wv.similar_by_word(word,10)

def cluster(word):
	test_words = [word]

	Test = vectorizer.transform(test_words)
 
	true_test_labels = ['Cricket','AI','Chemistry']
	predicted_labels_knn = modelknn.predict(Test)

	return true_test_labels[np.int(predicted_labels_knn[0])]















