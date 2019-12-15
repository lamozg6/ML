import flask
from flask import request, jsonify
import w2v

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Distant Reading Archive</h1>
<p>A prototype API for distant reading of science fiction novels.</p>'''

@app.route('/similar', methods=['GET'])
def api_similar_words():
    if 'word' not in request.args:
        return "Error: No word provided. Please specify a word."

    return jsonify(w2v.similar_words(request.args['word']))

@app.route('/cluster', methods=['GET'])
def api_cluster():
    if 'word' not in request.args:
        return "Error: No word provided. Please specify a word."

    return jsonify(w2v.cluster(request.args['word']))

app.run()
