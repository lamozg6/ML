from nltk.corpus import stopwords
import numpy as np
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import argparse
from sklearn.cluster import KMeans

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--window_size', type = int, default = 5)
parser.add_argument('--vocabulary_size', type = int, default = 20)
parser.add_argument('--n_clusters', type = int, default = 4)
parser.add_argument('--inf', type = str, default = "in.txt")
parser.add_argument('--outf', type = str, default = "out.txt")

args = parser.parse_args()
w = args.window_size
x = args.vocabulary_size

# -----------------------------------------------------------------------------
 
def clean(doc):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y

def find_element_in_list(element, lst):
    try:
        index_element = lst.index(element)
        return index_element
    except ValueError:
        return None
          
def word_vectors(words, base_vocabulary):
    bug_of_words = np.zeros(shape=(len(words), x), dtype = np.int32)

    for i in range(w, len(words) - w):
        for j in range (i - w, i + w + 1):
            if j != i:
                word_idx = find_element_in_list(words[j], base_vocabulary)
                if word_idx != None:
                    bug_of_words[i - w][word_idx] += 1
                    
    return bug_of_words

def vectors_to_file(words, bug_of_words):
    out = open(args.outf, "w")
    for idx in range(len(words)):
        out.write(words[idx] + ' -> ' + np.array2string(bug_of_words[idx]) + '\n')
    out.close()
    
def k_means(bug_of_words):
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(bug_of_words)
    print(kmeans.labels_)
    
if __name__ == '__main__':
    file = open(args.inf, "r") 
    data = file.read()
    file.close()

    words = clean(data)
    fdist = FreqDist(words)

    base_vocabulary = [word for (word, count) in fdist.most_common(x)]

    dummy = [' ' for x in range(w)]
    words = dummy + words + dummy
    
    bug_of_words = word_vectors(words, base_vocabulary)

    words = words[w:-w]
    
    vectors_to_file(words, bug_of_words)
    
    k_means(bug_of_words)