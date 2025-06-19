from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/sentiment', methods=['POST'])
def sentiment_analysis():
    try:
        data = request.json
        text = data['text']
        scores = polarity_scores_roberta(text)
        return jsonify(scores)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def polarity_scores_roberta(example):
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        # 'roberta_neg' : scores[0],
        # 'roberta_neu' : scores[1],
        # 'roberta_pos' : scores[2]
        'negative': float(scores[0]),
        'neutral': float(scores[1]),
        'positive': float(scores[2])
    }
    return scores_dict


if __name__ == '__main__':
    app.run(debug=True)
