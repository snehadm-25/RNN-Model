import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate():
    print("Loading test data and model...")
    X_test = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    with open('processed_data/label_encoder.pickle', 'rb') as handle:
        le = pickle.load(handle)
        
    model = load_model('best_model.keras')
    
    print("Generating predictions...")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save confusion matrix as an image
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate()
