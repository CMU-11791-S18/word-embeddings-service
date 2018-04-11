
from gensim.models import KeyedVectors
class SimilarityService

    def _init_(self):
        filename = 'GoogleNews-vectors-negative300.bin'
        self.model = KeyedVectors.load_word2vec_format(filename, binary=True)

    @static
    def get_word_vector(word):
        return self.model[word]


    @static
    def get_similar_word_embeddings(positive_words, negative_words, topn = 1):
        return self.model.most_similar(positive=positive_words, negative=negative_words, topn=topn)

    @static
    def get_word_to_word_similarity(first_word, second_word):
        return self.model.wv.similarity(first_word, second_word)
