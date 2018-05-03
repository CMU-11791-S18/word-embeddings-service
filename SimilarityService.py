from WordEmbeddingFactory import WordEmbeddingFactory


class SimilarityService:

    @classmethod
    def get_word_vector(cls, embedding, word):
        return WordEmbeddingFactory.get_model(embedding=embedding).get_word_vector(word=word)

    @classmethod
    def get_similar_word_embeddings(cls, embedding, positive_words, negative_words, topn = 1):
        return WordEmbeddingFactory.get_model(embedding=embedding).get_similar_word_embeddings(positive_words=positive_words, negative_words=negative_words, topn=topn)

    @classmethod
    def get_word_to_word_similarity(cls, embedding, first_word, second_word):
        return WordEmbeddingFactory.get_model(embedding=embedding).get_word_to_word_similarity(first_word=first_word, second_word=second_word)

    @classmethod
    def get_sentence_similarity_matrix(cls, embedding, s1, s2):
        return WordEmbeddingFactory.get_model(embedding=embedding).get_sentence_similarity(s1=s1, s2=s2)
