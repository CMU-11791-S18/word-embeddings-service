import datetime
import time
from logging.handlers import RotatingFileHandler

from flask import Flask, logging
from flask import Response
from flask import request
from SimilarityService import SimilarityService


app = Flask(__name__)
formatter = logging.Formatter(
        "[%(levelname)s - %(message)s]"
)
logFileName = 'logs/{}.log'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M'))
handler = RotatingFileHandler(logFileName, maxBytes=10000000)
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
app.logger.addHandler(handler)
log = logging.getLogger('werkzeug')
log.setLevel(logging.DEBUG)
log.addHandler(handler)

print("Started Word Embedding service on port 80...")


@app.route('/getWordVector')
def get_word_embeddings():
    try:
        # app.logger.info(request.method, request.path, request.args)
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
        # app.logger.info(request.method, request.path, request.args)
        positive_words = request.args.get('positive_words').split(',')
        negative_words = []
        topn = 1
        if request.args.get('negative_words') is not None:
            negative_words = request.args.get('negative_words').split(',')
        if request.args.get('topn') is not None:
            topn = int(request.args.get('topn'))
        return Response(SimilarityService.get_similar_word_embeddings(positive_words, negative_words, topn),
                        status=200,
                        content_type='application/json')

    except KeyError:
        content = {'message': 'INVALID PARAMS'}
        return Response(content, status=400, content_type='application/json')


@app.route('/getWordToWordSimilarity')
def get_word_to_word_similarity():
    try:
        # app.logger.info(request.method, request.path, request.args)
        word1 = request.args.get('word1')
        word2 = request.args.get('word2')
        return Response(SimilarityService.get_word_to_word_similarity(word1, word2),
                        status=200,
                        content_type='application/json')

    except KeyError:
        content = {'message': 'INVALID PARAMS'}
        return Response(content, status=400, content_type='application/json')


@app.route('/getSentenceSimilarityMatrix')
def get_sentence_similarity():
    try:
        # app.logger.info(request.method, request.path, request.args)
        s1 = request.args.get('s1')
        s2 = request.args.get('s2')
        return Response(SimilarityService.get_sentence_similarity_matrix(s1, s2),
                        status=200,
                        content_type='application/json')
    except KeyError:
        content = {'message': 'INVALID PARAMS'}
        return Response(content, status=400, content_type='application/json')


@app.after_request
def after_request(response):
    app.logger.info(request.method, request.path, request.args)
    return response


if __name__ == '__main__':
    formatter = logging.Formatter(
        "[%(levelname)s - %(message)s]"
    )
    logFileName = 'logs/{}.log'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M'))
    handler = RotatingFileHandler(logFileName, maxBytes=10000000)
    handler.formatter = formatter
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.DEBUG)
    log.addHandler(handler)
    app.run(threaded=True)
