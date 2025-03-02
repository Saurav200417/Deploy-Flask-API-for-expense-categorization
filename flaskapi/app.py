from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model and preprocessing tools
model = load_model('expense_classifier.h5')
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    # Preprocess input
    features = vectorizer.transform([text]).toarray()
    prediction = model.predict(features)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])

    return jsonify({'category': predicted_class[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))