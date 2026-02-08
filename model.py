import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, SpatialDropout1D, Bidirectional, GlobalMaxPooling1D, BatchNormalization
from tensorflow.keras.regularizers import l2

def build_model(vocab_size, num_classes, max_len=1000):
    model = Sequential([
        Embedding(vocab_size, 128),
        SpatialDropout1D(0.4),
        Bidirectional(GRU(64, return_sequences=True)),
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Test building the model
    model = build_model(10000, 25) # Example values
    model.summary()
