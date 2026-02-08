import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text

def predict_category(resume_text):
    # Load artifacts
    model = load_model('best_model.keras')
    with open('processed_data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('processed_data/label_encoder.pickle', 'rb') as handle:
        le = pickle.load(handle)
    
    # Preprocess the input text
    cleaned = clean_text(resume_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=500, padding='post', truncating='post')
    
    # Predict
    pred_prob = model.predict(padded)
    pred_idx = np.argmax(pred_prob, axis=1)[0]
    confidence = pred_prob[0][pred_idx]
    
    category = le.inverse_transform([pred_idx])[0]
    
    return category, confidence

if __name__ == "__main__":
    sample_resume = """
    Experienced Software Engineer with a strong background in developing scalable web applications.
    Proficient in Python, Java, and JavaScript. Experienced with cloud platforms like AWS and Azure.
    Developed multiple machine learning models using TensorFlow and PyTorch.
    """
    
    print("Sample Resume Text:")
    print("-" * 20)
    print(sample_resume.strip())
    print("-" * 20)
    
    print("\nPredicting category...")
    category, confidence = predict_category(sample_resume)
    print(f"Predicted Category: {category} (Confidence: {confidence:.2f})")
