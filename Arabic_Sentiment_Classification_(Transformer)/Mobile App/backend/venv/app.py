from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Text is not provided"})
    
    model_path = "models/checkpoint-90"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")

    # Initialize the pipeline for text classification on GPU
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

    sentiment = classifier(text)[0]

    labels_map = {
        "LABEL_0": "Positive",
        "LABEL_1": "Neutral",
        "LABEL_2": "Negative"
    }

    sentiment['label'] = labels_map.get(sentiment['label'], sentiment['label'])
    sentiment["score"] = np.round(sentiment["score"], 2)
    
    return jsonify({
        "text": text,
        "sentiment": sentiment
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


#  curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text": "This project is good"}'