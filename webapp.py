import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text

app = Flask(__name__)
CORS(app)

# Force high-accuracy settings
MAX_LEN = 1000

# Load model and artifacts
print("Loading model and artifacts...")
model = tf.keras.models.load_model('best_model.keras')

with open('processed_data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('processed_data/label_encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'resume_text' not in data:
            return jsonify({'error': 'No resume text provided'}), 400
        
        resume_text = data['resume_text']
        
        # Preprocess
        cleaned_text = clean_text(resume_text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Predict
        prediction = model.predict(padded_sequence)
        class_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        category = le.inverse_transform([class_idx])[0]
        
        # Get top 3 categories
        top_3_idx = np.argsort(prediction[0])[-3:][::-1]
        top_3 = []
        for idx in top_3_idx:
            top_3.append({
                'category': le.inverse_transform([idx])[0],
                'confidence': float(prediction[0][idx])
            })

        return jsonify({
            'category': category,
            'confidence': confidence,
            'top_3': top_3
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use port 5000 by default
    app.run(host='0.0.0.0', port=5000, debug=True)
