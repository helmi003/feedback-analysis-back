from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from flask_cors import CORS
import torch
import psutil
import os
import time

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

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

if torch.cuda.is_available():
    model = model.to('cuda')
    print("[INFO] Model loaded on GPU")
else:
    print("[INFO] Model running on CPU")

def polarity_scores_roberta(example):
    start = time.perf_counter()
    with torch.no_grad():
        encoded_text = tokenizer(example, return_tensors='pt')
        if torch.cuda.is_available():
            encoded_text = {k: v.to('cuda') for k, v in encoded_text.items()}
        output = model(**encoded_text)
        scores = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)
        scores_dict = {
            'Negative': round(float(scores[0]) * 100, 2),
            'Neutral': round(float(scores[1]) * 100, 2),
            'Positive': round(float(scores[2]) * 100, 2),
        }
    end = time.perf_counter()
    print(f"[INFERENCE TIME] {end - start:.3f} seconds")
    log_system_usage()
    log_gpu_usage()
    return scores_dict

def log_system_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    cpu = process.cpu_percent(interval=0.1)
    print(f"[SYSTEM USAGE] Memory: {mem_mb:.2f} MB | CPU: {cpu:.2f}%")

def log_gpu_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"[GPU USAGE] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")


if __name__ == '__main__':
    app.run(debug=True)
