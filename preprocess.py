import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)  # remove URLs
    text = re.sub(r'RT|cc', ' ', text)  # remove RT and cc
    text = re.sub(r'#\S+', ' ', text)  # remove hashtags
    text = re.sub(r'@\S+', ' ', text)  # remove mentions
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)  # remove punctuations
    text = re.sub(r'[^\x00-\x7f]', r' ', text) 
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    return " ".join(cleaned_words)

def preprocess_data(csv_path):
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Use Resume_str for text and Category for labels
    print("Cleaning text...")
    df['cleaned_resume'] = df['Resume_str'].apply(clean_text)
    
    print("Encoding labels...")
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['Category'])
    
    print("Tokenizing...")
    max_words = 20000
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['cleaned_resume'])
    
    sequences = tokenizer.texts_to_sequences(df['cleaned_resume'])
    
    max_len = 1000 # Increased length to cover more of the resume
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    X = padded_sequences
    y = df['category_encoded'].values
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save artifacts for training and evaluation
    os.makedirs('processed_data', exist_ok=True)
    np.save('processed_data/X_train.npy', X_train)
    np.save('processed_data/X_test.npy', X_test)
    np.save('processed_data/y_train.npy', y_train)
    np.save('processed_data/y_test.npy', y_test)
    
    with open('processed_data/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('processed_data/label_encoder.pickle', 'wb') as handle:
        pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Preprocessing complete. Artifacts saved in 'processed_data/'")
    return X_train, X_test, y_train, y_test, le

if __name__ == "__main__":
    preprocess_data('Resume/Resume.csv')
