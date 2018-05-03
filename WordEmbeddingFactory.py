from Word2VecEmbeddingModel import Word2VecEmbeddingModel


class WordEmbeddingFactory:

    @classmethod
    def get_model(cls, embedding):
        return Word2VecEmbeddingModel.instance()
