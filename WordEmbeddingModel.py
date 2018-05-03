from abc import abstractmethod


class WordEmbeddingModel:

    @abstractmethod
    def get_word_vector(self, word):
        pass

    @abstractmethod
    def get_similar_word_embeddings(self, positive_words, negative_words, topn=1):
        pass

    @abstractmethod
    def get_word_to_word_similarity(self, first_word, second_word):
        pass

    @abstractmethod
    def get_sentence_similarity(self, s1, s2):
        pass
