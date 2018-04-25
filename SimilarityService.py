import os

from gensim.models import KeyedVectors


class SimilarityService:

    # filename = 'PubMed-w2v.bin'
    filename = os.getenv('WORD2VEC_BINARY_FILE')
    model = KeyedVectors.load_word2vec_format(filename, binary=True)

    @classmethod
    def get_word_vector(self, word):
        return str(self.model[word])

    @classmethod
    def get_similar_word_embeddings(self, positive_words, negative_words, topn = 1):
        return str(self.model.most_similar(positive=positive_words, negative=negative_words, topn=topn))

    @classmethod
    def get_word_to_word_similarity(self, first_word, second_word):
        return str(self.model.wv.similarity(first_word, second_word))

    @classmethod
    def get_sentence_similarity_matrix(self, s1, s2):
        m, n = len(set1), len(set2)
        W_s1_s2 = [[self.getWordToWordSimilarity(s1[x], s2[y]) for y in range(n)] for x in range(m)]
        return W_s1_s2
