import random
from index import Index
from document import Document
from scipy.sparse import load_npz
from scipy.spatial.distance import cosine
from preprocess_functions import text_preprocess
from encoders import *

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
    
def encode_text(text: str):
    # normilize text
    text = text_preprocess(text)

    # word2vec encoding of the text
    tmp = np.asarray([text_to_vec(text)])
    w2v = np.concatenate([tmp, tmp, tmp], axis=1)
    
    # tfidf encoding of the text
    text = np.asarray([text])
    tfidf = np.hstack([
        for_title.transform(text).toarray(),
        for_subTitle.transform(text).toarray(),
        for_text.transform(text).toarray(),
    ])
    
    return tfidf, w2v

# returns indices of documents which contain words of the query
def build_indices(query: str):
    indices = np.asarray([], dtype=int)
    while True:
        new = np.asarray(list(set(index.documents_for_query(query)).difference(indices)))
        indices = np.hstack([indices, new])
        # in case there are too few documents - shorten the query and try again
        if len(indices) < 20 and len(query) > 0:
            query = query.rsplit(' ', 1)[0] 
        else:
            break

    # if still there are not enough documents - return random indices
    if len(indices) < 20:
        others = list(set(df.index).difference(indices))
        np.hstack([indices, np.random.choice(others, 10 - len(indices))])
        
    return indices.astype(int)

# how much a document corresponds to a query
def score(query: str, document: str):
    query_tfidf, query_w2v = encode_text(query)
    doc_tfidf, doc_w2v = encode_text(document)

    return 0.3 * cs(doc_tfidf, query_tfidf) \
            + 0.7 * cs(doc_w2v, query_w2v)

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
    indices = build_indices(query)

    query_tfidf, query_w2v = encode_text(query)
    # take a subset of documents
    sub_df_tfidf = pd.DataFrame(tfidf_df[indices].toarray())
    sub_df_w2v = w2v_df.iloc[indices, :]
    
    # calculate similarities
    sims = 0.3 * sub_df_tfidf.apply(lambda x: cs(x, query_tfidf), axis=1) \
        + 0.7 * sub_df_w2v.apply(lambda x: cs(x, query_w2v), axis=1)
    
    if not sims.empty:
        top = sims.sort_values(ascending = False).head(20)
    else:
        top = df.head(20)
    return df_to_docs(df.iloc[top.index, :])
