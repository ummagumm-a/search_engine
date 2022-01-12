import random
from index import Index
from document import Document
from scipy.sparse import load_npz
from preprocess_functions import text_preprocess

from utils import *
from encoders import *

import pandas as pd
import numpy as np

index = None
df = None
tfidf_df = None
w2v_df = None

def init_dfs():
    global df, tfidf_df, w2v_df
    df = pd.read_csv('data/medium.csv')
    tfidf_df = load_npz('data/tfidf_df.npz')
    w2v_df = pd.read_csv('data/w2v_df.csv')

def build_index():
    global index
    prep_df = pd.read_csv('data/prep_df.csv').applymap(str)
    index = Index(prep_df['text'].tolist())

# returns indices of documents which contain words of the query
def build_indices(query: str):
    indices = index.documents_for_query(query)
    # in case there are too few documents - shorten the query and try again
    while len(indices) < 20 and len(query) > 0:
        query = query.rsplit(' ', 1)[0]
        new = np.asarray(list(set(index.documents_for_query(query)).difference(indices)))
        indices = np.hstack([indices, new])

    # if still there are not enough documents - return random indices
    if len(indices) < 100:
        others = list(set(df.index).difference(indices))
        np.hstack([indices, np.random.choice(others, 100 - len(indices))])
    # if too many documents - pick top 100
    elif len(indices) > 100:
        indices = df['recommends'][indices].sort_values(ascending=False).index[:100]

    return indices.astype(int)

# how much a document corresponds to a query
def score(query: str, document: Document):
    query_tfidf, query_w2v = encode_query(query)
    doc_tfidf, doc_w2v = document.encoded()

    return score_with_encoded(query_tfidf, query_w2v, doc_tfidf, doc_w2v)

# score of an encoded query and encoded document (in vector form so far)
def score_with_vec(query_e, vec):
    shape = get_shape(query_e)
    doc_e = split_vec(vec, shape)

    return score_halved(query_e, doc_e)

# calculate score as a weighted sum of cosine similarities between
# title, subTitle and text vectors
def score_halved(query_e, doc_e):
    return (0.3 * cs(doc_e[0], query_e[0]) \
          + 0.2 * cs(doc_e[1], query_e[1]) \
          + 0.5 * cs(doc_e[2], query_e[2])) \

# score as a weighted sum of scores in tfidf and w2v encodings
def score_with_encoded(query_tfidf, query_w2v, doc_tfidf, doc_w2v):
    return 0.3 * score_halved(query_tfidf, doc_tfidf) \
         + 0.7 * score_halved(query_w2v, doc_w2v) \

# retrieve documents corresponding to a query
def retrieve(query: str):
    # indices of documents with words from query
    if query == '':
        top = df.head(10)
    else:
        indices = build_indices(query)

        query_tfidf, query_w2v = encode_query(query)
        # take a subset of documents
        sub_df_tfidf = pd.DataFrame(tfidf_df[indices].toarray())
        sub_df_w2v = w2v_df.iloc[indices, :]
        
        # calculate similarities
        sims = pd.Series(
            0.3 * sub_df_tfidf.apply(lambda x: score_with_vec(query_tfidf, x), axis=1).to_numpy() \
            + 0.7 * sub_df_w2v.apply(lambda x: score_with_vec(query_w2v, x), axis=1).to_numpy(),
            index=indices)
    
        top = sims.sort_values(ascending = False).head(10)

    return df_to_docs(df.iloc[top.index, :])

