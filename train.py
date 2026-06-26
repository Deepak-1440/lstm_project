import os
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import nltk

def download_data():
    print("--- Step 1: Downloading and Loading Shakespeare's Hamlet dataset ---")
    try:
        nltk.download('gutenberg')
        from nltk.corpus import gutenberg
        data = gutenberg.raw('shakespeare-hamlet.txt')
        print("Successfully loaded Hamlet from NLTK gutenberg corpus.")
    except Exception as e:
        print(f"Error loading from NLTK: {e}")
        print("Using fallback Shakespeare text...")
        # A fallback subset of Hamlet/Shakespeare lines to ensure training works offline
        data = """
        To be, or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles
        And by opposing end them. To die—to sleep,
        No more; and by a sleep to say we end
        The heart-ache and the thousand natural shocks
        That flesh is heir to: 'tis a consummation
        Devoutly to be wish'd. To die, to sleep;
        To sleep, perchance to dream—ay, there's the rub:
        For in that sleep of death what dreams may come,
        When we have shuffled off this mortal coil,
        Must give us pause. There's the respect
        That makes calamity of so long life.
        """
    
    # Save raw data to hamlet.txt in case it's needed
    with open('hamlet.txt', 'w', encoding='utf-8') as f:
        f.write(data)
    return data

def preprocess_and_tokenize(text):
    print("\n--- Step 2: Preprocessing and Tokenizing Text ---")
    text = text.lower()
    
    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    print(f"Vocabulary size (total words): {total_words}")
    
    # Save the tokenizer using pickle
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved tokenizer.pickle")
    
    return tokenizer, total_words, text

def create_sequences(tokenizer, text, max_sequence_len=16):
    print("\n--- Step 3: Creating N-gram Sequences ---")
    input_sequences = []
    
    # 1) Line-level n-grams
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
            
    # 2) Sliding-window n-grams (WINDOW_SIZE = 12)
    WINDOW_SIZE = 12
    all_tokens = tokenizer.texts_to_sequences([text.replace('\n', ' ')])[0]
    for i in range(1, len(all_tokens)):
        start = max(0, i - WINDOW_SIZE)
        n_gram_sequence = all_tokens[start:i + 1]
        if len(n_gram_sequence) > 1:
            input_sequences.append(n_gram_sequence)
            
    print(f"Total sequences generated: {len(input_sequences)}")
    
    # Cap sequences to max_sequence_len and pad
    input_sequences = [seq[-max_sequence_len:] if len(seq) > max_sequence_len else seq for seq in input_sequences]
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    # Predictors (x) and labels (y)
    x, y = input_sequences[:, :-1], input_sequences[:, -1]
    
    # Convert labels to one-hot encoding
    y = tf.keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index) + 1)
    
    # Split into train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test

def build_model(total_words, max_sequence_len):
    print("\n--- Step 4: Building and Compiling LSTM Model ---")
    model = Sequential()
    
    # Optimized architecture for fast CPU/GPU training (recurrent_dropout=0)
    model.add(Embedding(total_words, 128, mask_zero=True))
    model.add(LSTM(150, return_sequences=True, dropout=0.2, recurrent_dropout=0.0))
    model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.0))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(total_words, activation='softmax'))
    
    model.build(input_shape=(None, max_sequence_len - 1))
    
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    loss_fn = CategoricalCrossentropy(label_smoothing=0.05)
    
    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def main():
    max_sequence_len = 16
    
    # Download dataset
    text_data = download_data()
    
    # Preprocess & Tokenize
    tokenizer, total_words, cleaned_text = preprocess_and_tokenize(text_data)
    
    # Save configurations
    config = {
        "max_sequence_len": max_sequence_len,
        "total_words": total_words
    }
    with open('config.json', 'w') as f:
        json.dump(config, f)
    print("Saved config.json")
    
    # Create input/output pairs
    x_train, x_test, y_train, y_test = create_sequences(tokenizer, cleaned_text, max_sequence_len)
    
    # Build LSTM Model
    model = build_model(total_words, max_sequence_len)
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    
    print("\n--- Step 5: Training Model ---")
    # Using 10 epochs for fast verification. Feel free to increase epochs for higher accuracy.
    epochs = 10
    batch_size = 64
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    
    print("\n--- Step 6: Saving Trained Model ---")
    model.save('model.h5')
    print("Saved model as model.h5")
    
    print("\nTraining completed successfully! Ready for Streamlit application.")

if __name__ == '__main__':
    main()
