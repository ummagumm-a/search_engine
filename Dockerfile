FROM python:3.8-slim-buster

RUN pip install pandas numpy sklearn scipy flask pickle-mixin jupyterlab nltk
RUN  apt-get update \
  && apt-get install -y wget unzip\
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN wget 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'
RUN unzip 'crawl-300d-2M.vec.zip' -d ./data
RUN rm 'crawl-300d-2M.vec.zip'

#RUN kaggle datasets download -d aiswaryaramachandran/medium-articles-with-content
RUN unzip ./data/medium-articles-with-content.zip -d ./data
RUN rm ./data/medium-articles-with-content.zip

RUN python3 -c "import nltk; nltk.download('omw-1.4'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
RUN jupyter nbconvert --to notebook --execute notebooks/dataset_prep.ipynb
RUN jupyter nbconvert --to notebook --execute notebooks/preprocess.ipynb
RUN jupyter nbconvert --to notebook --execute notebooks/ml_part.ipynb

CMD [ "python3", "server.py" ]
