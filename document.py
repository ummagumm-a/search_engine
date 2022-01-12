from preprocess_functions import text_preprocess
from encoders import *

class Document:
    def __init__(self, title, subTitle, text, recommends):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.subTitle = subTitle
        self.text = text
        self.recommends = recommends
    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text[:500] + ' ...']

    def normalized(self):
        title = text_preprocess(self.title)
        subTitle = text_preprocess(self.subTitle)
        text = text_preprocess(self.text)

        return Document(title, subTitle, text, self.recommends)

    def encoded(self):
        return self.tfidf_encoded(), \
               self.w2v_encoded()

    def tfidf_encoded(self):
        doc = self.normalized()

        tfidf = (
            for_title.transform(np.asarray([doc.title])).toarray(),
            for_subTitle.transform(np.asarray([doc.subTitle])).toarray(),
            for_text.transform(np.asarray([doc.text])).toarray()
        )

        return tfidf

    def w2v_encoded(self):
        doc = self.normalized()

        w2v = (
            np.asarray([text_to_vec(doc.title)]),
            np.asarray([text_to_vec(doc.subTitle)]),
            np.asarray([text_to_vec(doc.text)]),
        )
        
        return w2v
