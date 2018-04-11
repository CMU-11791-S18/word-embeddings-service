from flask import Flask
from flask.ext.api import status
import SimilarityService from SimilarityService

app = Flask(__name__)

@app.route('/getWordVector')
def get_word_embeddings():
    try:
        word = request.args.get('word')
        return SimilarityService.get_word_vector(word)
    except KeyError:
        contect = { 'message' : 'INVALID PARAMS' }
        return content, status.HTTP_400_BAD_REQUEST

@app.route('/getSimilarWordEmbeddings')
def get_similar_word_embeddings():
    try:
        positive_words = request.args.get('positive_words').split(',')
        negative_words=[]
        topn = 1
        if request.args.get('negative_words') is not None:
            negative_words = request.args.get('negative_words').split(',')
        if request.args.get('topn') is not None:
            topn = request.args.get('topn')
        return SimilarityService.get_similar_word_embeddings(positive_words, negative_words , topn)

    except KeyError:
        contect = { 'message' : 'INVALID PARAMS' }
        return content, status.HTTP_400_BAD_REQUEST


@app.route('/getWordToWordSimilarity')
def getWordToWordSimilarity():
    try:
        word1 = request.args.get('word1')
        word2 = request.args.get('word2')
        return SimilarityService.get_similar_word_embeddings(word1,word2)

    except KeyError:
        contect = { 'message' : 'INVALID PARAMS' }
        return content, status.HTTP_400_BAD_REQUEST


@app.route('/getSentenceSimilarityMatrix')
def getSentenceSimilarityMatrix():
    return "Sentence"

if __name__ == '__main__':
    app.run(threaded=True, debug=True)
    SimilarityService()
