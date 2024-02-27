import config
import flask
from flask import Flask, request, render_template
import json
import nltk
nltk.download('punkt')

from simpletransformers.classification import ClassificationModel, ClassificationArgs
app = Flask(__name__)


def sentence_pair_prediction(sentence_pair):
    # tokenizer = config.TOKENIZER
    # max_len = config.MAX_LEN
    #
    # review = str(sentence)
    # review = ' '.join(review.split())
    # inputs = tokenizer.encode_plus(review, max_length=max_len, pad_to_max_length=True)
    #
    # ids = torch.LongTensor(inputs['input_ids']).unsqueeze(0).to(DEVICE)
    # mask = torch.LongTensor(inputs['attention_mask']).unsqueeze(0).to(DEVICE)
    # token_type_ids = torch.LongTensor(inputs['token_type_ids']).unsqueeze(0).to(DEVICE)
    #
    # logits = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)
    # outputs = torch.sigmoid(logits).cpu().detach().numpy()
    outputs = MODEL.predict(sentence_pair)

    return outputs[0][0]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    sentence = request.json['sentence']
    if sentence != '':
        positive = sentence_pair_prediction(nltk.sent_tokenize(sentence))
        print(nltk.sent_tokenize(sentence))
        print(positive)
        negative = 1 - positive
        response = {}
        response['response'] = {
            'positive': f'{positive:.4f}',
            'negative': f'{negative:.4f}',
            'sentence': str(sentence),
            'result': 'positive' if positive > 0.5 else 'negative'
        }
        return flask.jsonify(response)
    else:
        res = dict({'message': 'Empty review'})
        return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')
    # except Exception as ex:
    #     res = dict({'message': str(ex)})
    #     return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')


if __name__ == '__main__':
    # MODEL = BERTBaseUncased()
    # MODEL.load_state_dict(torch.load(config.MODEL_PATH, map_location=DEVICE))
    # MODEL.to(DEVICE)
    # MODEL.eval()
    MODEL = ClassificationModel(
        "bert", 'C:\\Users\\USER\\PycharmProjects\\pythonProject_practice\\best_model',
        use_cuda=False
    )
    app.run(debug=True, use_reloader=True)
