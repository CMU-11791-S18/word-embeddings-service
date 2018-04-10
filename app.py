from flask import Flask

app = Flask(__name__)


@app.route('/getWordEmbeddings')
def getWordEmbeddings():
    return "Word";


@app.route('/getWordToWordSimilarity')
def getWordToWordSimilarity():
    return "WordToWord"


@app.route('/getSentenceSimilarityMatrix')
def getSentenceSimilarityMatrix():
    return "Sentence"


if __name__ == '__main__':
    app.run(threaded=True, debug=True)
