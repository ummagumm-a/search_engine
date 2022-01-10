class Document:
    def __init__(self, title, subTitle, text, recommends):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.subTitle = subTitle
        self.text = text
        self.recommends = recommends
    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text[:250] + ' ...']
