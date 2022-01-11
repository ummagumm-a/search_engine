import random
from index import Index
from document import Document
from scipy.sparse import load_npz
from scipy.spatial.distance import cosine
from preprocess_functions import text_preprocess

from utils import *

import pandas as pd
import numpy as np

def build_index():
    prep_df = pd.read_csv('data/prep_df.csv').applymap(str)
    return Index(prep_df['text'].tolist())

index = build_index()
df = pd.read_csv('data/medium.csv')
tfidf_df = load_npz('data/tfidf_df.npz')
w2v_df = pd.read_csv('data/w2v_df.csv')

def cs(x, y):
    return 1 - cosine(x, y)
    
def encode_query(query: str):
    return tfidf_encode_query(query), \
           w2v_encode_query(query)

def tfidf_encode_query(query: str):
    return Document(query, query, query, 0).tfidf_encoded()

def w2v_encode_query(query: str):
    return Document(query, query, query, 0).w2v_encoded()

# returns indices of documents which contain words of the query
def build_indices(query: str):
    indices = np.asarray([], dtype=int)
    while True:
        print('build_indices: ')
        new = np.asarray(list(set(index.documents_for_query(query)).difference(indices)))
        indices = np.hstack([indices, new])
        # in case there are too few documents - shorten the query and try again
        if len(indices) < 20 and len(query) > 0:
            query = query.rsplit(' ', 1)[0] 
        else:
            break

    # if still there are not enough documents - return random indices
    if len(indices) < 20:
        print('not enough')
        others = list(set(df.index).difference(indices))
        np.hstack([indices, np.random.choice(others, 20 - len(indices))])
        
    if len(indices) > 100:
        print('too much')
        indices = df['recommends'][indices].sort_values(ascending=False).index[:100]

    return np.sort(indices.astype(int))

# how much a document corresponds to a query
def score(query: str, document: Document):
    query_tfidf, query_w2v = encode_query(query)
    doc_tfidf, doc_w2v = document.encoded()

    return score_with_encoded(query_tfidf, query_w2v, doc_tfidf, doc_w2v)

def score_with_vec(query_e, vec):
    shape = get_shape(query_e)
    doc_e = split_vec(vec, shape)

    return score_halved(query_e, doc_e)

def score_halved(query_e, doc_e):
    return (0.5 * cs(doc_e[0], query_e[0]) \
          + 0.1 * cs(doc_e[1], query_e[1]) \
          + 0.4 * cs(doc_e[2], query_e[2])) \

def score_with_encoded(query_tfidf, query_w2v, doc_tfidf, doc_w2v):
    return 0.3 * score_halved(query_tfidf, doc_tfidf) \
         + 0.7 * score_halved(query_w2v, doc_w2v) \

# transform dataframe row into Document object
def row_to_docs(row):
    return Document(
        row.title,
        row.subTitle,
        row.text,
        row.recommends)

# transform dataframe into a list of Documents
def df_to_docs(df):
    return df.apply(row_to_docs, axis=1).tolist()

# retrieve documents corresponding to a query
def retrieve(query: str):
    # indices of documents with words from query
    if query == '':
        top = df.head(20)
    else:
        print(1)
        indices = build_indices(query)
        print(indices)

        print(2)
        query_tfidf, query_w2v = encode_query(query)
        # take a subset of documents
        print(2.5)
        sub_df_tfidf = pd.DataFrame(tfidf_df[indices].toarray())
        sub_df_w2v = w2v_df.iloc[indices, :]
        print(3)
        
        # calculate similarities
        sims = pd.Series(
            0.3 * sub_df_tfidf.apply(lambda x: score_halved(query_tfidf, x), axis=1) \
            + 0.7 * sub_df_w2v.apply(lambda x: score_halved(query_w2v, x), axis=1),
            index=indices)
        print(4)
    
        top = sims.sort_values(ascending = False).head(20)
        print('top:')
        print(top.index)
        print(5)

    return df_to_docs(df.iloc[top.index, :])

