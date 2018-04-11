from flask import Flask
from flask import Response
from requests import request
from SimilarityService import SimilarityService

app = Flask(__name__)

@app.route('/getWordVector')
def get_word_embeddings():
    try:
        word = request.args.get('word')
        return Response(SimilarityService.get_word_vector(word),
                        status=200,
                        content_type='application/json')
    except KeyError:
        content = {'message': 'INVALID PARAMS'}
        return Response(content, status=400, content_type='application/json')


@app.route('/getSimilarWordEmbeddings')
def get_similar_word_embeddings():
    try:
        positive_words = request.args.get('positive_words').split(',')
        negative_words = []
        topn = 1
        if request.args.get('negative_words') is not None:
            negative_words = request.args.get('negative_words').split(',')
        if request.args.get('topn') is not None:
            topn = request.args.get('topn')
        return Response(SimilarityService.get_similar_word_embeddings(positive_words, negative_words, topn),
                        status=200,
                        content_type='application/json')

    except KeyError:
        content = {'message': 'INVALID PARAMS'}
        return Response(content, status=400, content_type='application/json')


@app.route('/getWordToWordSimilarity')
def get_word_to_word_similarity():
    try:
        word1 = request.args.get('word1')
        word2 = request.args.get('word2')
        return Response(SimilarityService.get_similar_word_embeddings(word1,word2),
                        status=200,
                        content_type='application/json')

    except KeyError:
        content = {'message': 'INVALID PARAMS'}
        return Response(content, status=400, content_type='application/json')


@app.route('/getSentenceSimilarityMatrix')
def get_sentence_similarity_():
    return SimilarityService.get_sentence_similarity_matrix(s1, s2)


if __name__ == '__main__':
    app.run(threaded=True, debug=True)
    SimilarityService()
