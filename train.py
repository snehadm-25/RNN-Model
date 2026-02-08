import numpy as np
import pickle
import os
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

def train():
    print("Loading processed data...")
    X_train = np.load('processed_data/X_train.npy')
    y_train = np.load('processed_data/y_train.npy')
    X_test = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    with open('processed_data/label_encoder.pickle', 'rb') as handle:
        le = pickle.load(handle)
        
    num_classes = len(le.classes_)
    vocab_size = 20000 
    
    print(f"Building model for {num_classes} classes...")
    model = build_model(vocab_size, num_classes)
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=100, # Increased for high-accuracy refinement with higher patience
        batch_size=64,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )
    
    print("Training complete. Model saved as 'best_model.keras'")
    
    # Save training history
    with open('processed_data/history.pickle', 'wb') as handle:
        pickle.dump(history.history, handle)

if __name__ == "__main__":
    train()
