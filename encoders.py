import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import numpy as np

# tfidf encoders
def load(name: str):
    with open('models/' + name + '.pickle', 'rb') as handle:
        model = pickle.load(handle)

    return model

for_title = load('for_title')
for_subTitle = load('for_subTitle')
for_text = load('for_text')

# word2vec encoding
def load_vectors(fname):
    with io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split())
        data = {}
        counter = 0
        for line in fin:
            counter+=1
            if counter == 100000:
                break
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(list(map(float, tokens[1:])))
            
    return data

vecs = load_vectors('data/crawl-300d-2M.vec')
zero = sum(vecs.values()) / len(vecs)

def text_to_vec(text):
    cum = np.zeros(300)
    words = text.split()
    
    if len(words) == 0:
        return zero
        
    for word in words:
        cum += vecs.get(word, zero)
        
    return cum / len(words)

