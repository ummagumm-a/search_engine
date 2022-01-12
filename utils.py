from document import Document
from scipy.spatial.distance import cosine

def get_shape(tpl):
    return list(map(lambda x: x.shape[1], tpl))

def split_vec(vec, splits):
    return (vec[:splits[0]],
            vec[splits[0]:splits[0] + splits[1]],
            vec[splits[0] + splits[1]:])

def cs(x, y):
    return 1 - cosine(x, y)

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

def encode_query(query: str):
    return tfidf_encode_query(query), \
           w2v_encode_query(query)

def tfidf_encode_query(query: str):
    return Document(query, query, query, 0).tfidf_encoded()

def w2v_encode_query(query: str):
    return Document(query, query, query, 0).w2v_encoded()

