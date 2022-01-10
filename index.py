import numpy as np

class Index():
    def __init__(self, documents: list):
        self.word_bank = {}
        for doc_i in range(len(documents)):
            for word in documents[doc_i].split():
                if not word in self.word_bank.keys():
                    self.word_bank[word] = []
                self.word_bank[word].append(doc_i)

    def documents_for_query(self, query: str):
        indices = []
        for word in query.split():
            if word in self.word_bank.keys():
                indices.append(self.word_bank[word])

        return self._intersect_indices(indices)

    def _intersect_indices(self, indices: list):
        n_indices = len(indices)
        if n_indices == 0:
            return []

        index_lens = np.asarray(list(map(len, indices)), dtype=int)
        pointers = np.zeros(n_indices, dtype=int)

        intersection = []
        while (index_lens > pointers).all():
            elements = np.asarray([i[p] for i, p in zip(indices, pointers)], dtype=int)
            if (elements == elements[0]).all():
                intersection.append(elements[0])
                pointers += 1
            else:
                pointers[np.argmin(elements)] += 1

        return np.asarray(intersection, dtype=int)

