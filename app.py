import copy
import csv
import os
import re
import string
import time

import numpy as np
from nltk import pos_tag, SnowballStemmer
from ruwordnet import RuWordNet

# import spacy_streamlit
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import flask
from flask import Flask, request, render_template, send_from_directory
import json
import nltk
# import ru_core_news_sm
# nltk.download('punkt')
import pymorphy2
from nltk.corpus import stopwords
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from process_sent_pairs import *
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = f'{app.root_path}/reports'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'sentence_from_file' in request.json:
        text = request.json['sentence_from_file']
        if text != '':
            sentences = nltk.sent_tokenize(text, language='russian')
            if len(sentences) > 1:
                sentences = [s.replace('\n', '') for s in sentences]
                tmp_report = []
                for i in range(1, len(sentences)):
                    tmp_preds = []
                    tmp_preds.append(float(MODEL.predict([[sentences[i - 1], sentences[i]]])[0]))

                    for p in predictors:
                        tmp_preds.append(p(sentences[i - 1], sentences[i]))

                    tmp_report.extend([[sentences[i - 1], sentences[i], *tmp_preds]])

                    preds.append(tmp_preds)
                preds_ = np.array(preds)
                reports.append(tmp_report)

                response = {}
                response['response'] = {
                    'mean_score': float("{:.5f}".format(np.mean(preds_[:, 0]))),
                    'mean_is_wordfrom_rep_in_sent_pairs': np.mean(preds_[:, 1]),
                    'mean_is_deriv_in_sent_pairs': np.mean(preds_[:, 2]),
                    'mean_is_hyponym_in_sent_pairs': np.mean(preds_[:, 3]),
                    'mean_is_hypernym_in_sent_pairs': np.mean(preds_[:, 4]),
                    'mean_is_anph_cand_in_sent_pairs': np.mean(preds_[:, 5])
                }
                preds.clear()
                return flask.jsonify(response)
            else:
                return flask.jsonify("Not enough sentences")
        else:
            res = dict({'message': 'Empty review'})
            return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')
    else:
        print('Please, choose file to process!')


@app.route('/download_csv/')
def download_csv():
    print(reports[-1])
    if len(reports[-1]) > 0:
        fn = f'report{time.strftime("%Y%m%d%H:%M:%S")}.csv'
        path_fn = f'reports/{fn}'
        with open(path_fn, 'w', newline='') as file:
            writer = csv.writer(file)
            for line in reports[-1]:
                writer.writerow(line)
        # report.clear()
        return send_from_directory(app.config['UPLOAD_FOLDER'], fn, as_attachment=True)
    else:
        print("No data for report!")
    return ('', 204)


if __name__ == '__main__':
    MODEL = ClassificationModel(
        "bert", '/best_model/',
        use_cuda=False
    )
    # models = ["ru_core_web_sm"]
    # default_text = "Москва столица России"
    # spacy_streamlit.visualize(models, default_text)
    app.run(debug=True, use_reloader=True)
