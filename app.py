import os
import pickle
import numpy as np
import tensorflow as tf
import gradio as gr
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text

# Configuration
MAX_LEN = 1000

# Load model and artifacts
print("Loading model and artifacts...")
model = tf.keras.models.load_model('best_model.keras')

with open('processed_data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('processed_data/label_encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)

def classify_resume(resume_text):
    if not resume_text:
        return "Please enter resume text.", None, []
    
    try:
        # Preprocess
        cleaned_text = clean_text(resume_text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Predict
        prediction = model.predict(padded_sequence)
        class_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        category = le.inverse_transform([class_idx])[0]
        
        # Format Top 3 for Gradio
        top_3_idx = np.argsort(prediction[0])[-3:][::-1]
        top_3_dict = {}
        for idx in top_3_idx:
            label = le.inverse_transform([idx])[0]
            top_3_dict[label] = float(prediction[0][idx])

        return category, confidence, top_3_dict
    
    except Exception as e:
        return f"Error: {str(e)}", 0, {}

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 AI Resume Classifier (RNN)")
    gr.Markdown("Experience high-accuracy resume categorization using a Bidirectional GRU model.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Paste Resume Text", placeholder="Copy and paste resume content here...", lines=10)
            btn = gr.Button("Classify Resume", variant="primary")
        
        with gr.Column():
            output_label = gr.Label(label="Predicted Category")
            output_conf = gr.Number(label="Overall Confidence Score")
            output_top3 = gr.Label(label="Alternative Matches (Top 3)")

    btn.click(fn=classify_resume, inputs=input_text, outputs=[output_label, output_conf, output_top3])
    
    gr.Examples(
        examples=[
            ["Project Manager with 10 years experience in construction, PMP certified, site management skills."],
            ["Software Engineer proficient in Python, TensorFlow, and AWS. Experienced in building scalable APIs."],
            ["Certified Public Accountant (CPA) with expertise in auditing, tax compliance, and financial reporting."]
        ],
        inputs=input_text
    )

if __name__ == "__main__":
    demo.launch()
