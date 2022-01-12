# search_engine
Homework for Tinkoff ML Course.

# Preview
![search_enginge_preview](https://github.com/ummagumm-a/search_engine/blob/master/search_engine_preview.gif)

# Data 
[Medium dataset](https://www.kaggle.com/aiswaryaramachandran/medium-articles-with-content)

# Prerequisites
* docker
* kaggle

# Setup
Clone the repo: <br />
`git clone git@github.com:ummagumm-a/search_engine.git` <br />
`cd search_engine` <br />
Download dataset (download exactly at 'data'!): <br />
`mkdir data && cd data` <br />
`kaggle datasets download -d aiswaryaramachandran/medium-articles-with-content` <br />
`cd ..` <br />
Docker part: <br />
`docker build -t search_engine .` <br />
`docker run -dp 8080:8080 search_engine` <br />
Wait for a minute until initialization is performed and then navigate to http://localhost:8080/.

# Implementation details:
* `dataset_prep` notebook extracts only useful columns from the dataset.
* `preprocess` notebook performs necessary preprocessing for all text in dataset such as puncuation and stop-words removal and lemmatization
* `ml_part` notebook encodes each row of dataset as a tf-idf vector and as a word2vec vector.
* Class `Index` is used to store indices of documents which contain a word. In such a way the model can choose only among relevant documents.
* Class `Document` stores records for each document. Also has a functionality to encode documents.
