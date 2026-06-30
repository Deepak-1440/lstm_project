import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import json
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Custom styling using Outfit font and dark-mode premium look
st.set_page_config(
    page_title="Shakespearean Next Word Predictor",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background-color: rgba(79, 70, 229, 0.05);
        border: 1px solid rgba(79, 70, 229, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .word-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #4f46e5;
        color: white;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-right: 0.5rem;
    }
    
    .prob-text {
        font-size: 0.9rem;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def load_assets():
    if not (os.path.exists('model.h5') and os.path.exists('tokenizer.pickle') and os.path.exists('config.json')):
        return None, None, None
        
    try:
        # Load keras model (model.h5)
        model = tf.keras.models.load_model('model.h5')
        
        # Load tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        # Load config
        with open('config.json', 'r') as f:
            config = json.load(f)
            
        return model, tokenizer, config
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None, None

# Main application logic
def main():
    st.markdown('<div class="main-header">✍️ Shakespearean Next Word Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">"a lightweight LSTM language model trained on Shakespeare's Hamlet that predicts and generates text in its style" \'s Hamlet</div>', unsafe_allow_html=True)
    
    model, tokenizer, config = load_assets()
    
    # Sidebar
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a2/Shakespeare.jpg", width=150)
    st.sidebar.header("Model Specifications")
    
    if model is not None:
        max_sequence_len = config.get("max_sequence_len", 16)
        total_words = config.get("total_words", 0)
        
        st.sidebar.success("Model loaded successfully!")
        st.sidebar.metric(label="Vocabulary Size", value=f"{total_words:,} words")
        st.sidebar.metric(label="Context Window Size", value=f"{max_sequence_len - 1} words")
        
        # Hyperparameters controls
        st.sidebar.markdown("---")
        st.sidebar.subheader("Generation Settings")
        
        temperature = st.sidebar.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic (arg max)."
        )
        
        top_k = st.sidebar.slider(
            "Suggestions shown",
            min_value=1,
            max_value=5,
            value=3,
            step=1
        )
        
        # Reset state button
        if st.sidebar.button("Clear Text Input"):
            st.session_state.user_input = ""
            st.rerun()
            
    else:
        st.sidebar.warning("Model files missing.")
        
    # Check if model files exist
    if model is None:
        st.warning("⚠️ Model assets (`model.h5`, `tokenizer.pickle`, `config.json`) could not be found.")
        st.info("To use this app, you need to train the model first. You can run training in the command line using:")
        st.code("python train.py", language="bash")
        
        st.markdown("### Or train the model directly from this Web App:")
        if st.button("🚀 Start Training Now (Takes ~1-2 minutes on CPU)"):
            with st.spinner("Training model... Please wait..."):
                try:
                    import train
                    # Execute training flow directly
                    progress_bar = st.progress(0)
                    st.text("Step 1: Downloading Gutenberg Corpus...")
                    text_data = train.download_data()
                    progress_bar.progress(20)
                    
                    st.text("Step 2: Preprocessing and Tokenizing...")
                    tokenizer, total_words, cleaned_text = train.preprocess_and_tokenize(text_data)
                    progress_bar.progress(40)
                    
                    # Save configurations
                    config = {
                        "max_sequence_len": 16,
                        "total_words": total_words
                    }
                    with open('config.json', 'w') as f:
                        json.dump(config, f)
                    
                    st.text("Step 3: Preparing Input/Output sequences...")
                    x_train, x_test, y_train, y_test = train.create_sequences(tokenizer, cleaned_text, 16)
                    progress_bar.progress(60)
                    
                    st.text("Step 4: Compiling model...")
                    m = train.build_model(total_words, 16)
                    progress_bar.progress(70)
                    
                    st.text("Step 5: Training model (10 Epochs)...")
                    # Redirect keras logs to streamlit or run fit with simple verbose
                    history = m.fit(
                        x_train, y_train,
                        epochs=10,
                        batch_size=64,
                        validation_data=(x_test, y_test),
                        verbose=1
                    )
                    progress_bar.progress(90)
                    
                    st.text("Step 6: Saving model assets...")
                    m.save('model.h5')
                    progress_bar.progress(100)
                    
                    st.success("🎉 Model trained and saved successfully! Refreshing app...")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Training failed: {ex}")
        return
        
    # App main area
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
        
    def append_word(word_to_append):
        current_text = st.session_state.user_input.strip()
        if current_text:
            st.session_state.user_input = current_text + " " + word_to_append + " "
        else:
            st.session_state.user_input = word_to_append + " "

    st.markdown("### Type your Shakespearean text below:")
    text_input = st.text_input(
        "Start typing here... (e.g., 'To be or not to')",
        value=st.session_state.user_input,
        key="user_input",
        placeholder="Type a word or phrase..."
    )
    
    # Process prediction
    cleaned_input = text_input.strip().lower()
    
    if cleaned_input:
        token_list = tokenizer.texts_to_sequences([cleaned_input])[0]
        max_sequence_len = config["max_sequence_len"]
        
        # Trim sequence if too long
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]
            
        # Pad sequence
        token_list_padded = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Predict probability distributions
        preds = model.predict(token_list_padded, verbose=0)[0]
        
        # Calculate temperature-adjusted predictions
        if temperature > 0:
            # Apply temperature scaling
            preds_adjusted = np.log(preds + 1e-10) / temperature
            exp_preds = np.exp(preds_adjusted)
            preds_adjusted = exp_preds / np.sum(exp_preds)
        else:
            # Deterministic (argmax equivalent)
            preds_adjusted = np.zeros_like(preds)
            preds_adjusted[np.argmax(preds)] = 1.0
            
        # Get top-k predictions
        top_indices = np.argsort(preds_adjusted)[::-1][:top_k]
        
        # Map indices to words and probabilities
        top_predictions = []
        for index in top_indices:
            word = ""
            for w, idx in tokenizer.word_index.items():
                if idx == index:
                    word = w
                    break
            if word:
                prob = preds[index] # Display actual probability
                top_predictions.append((word, prob))
        
        if top_predictions:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("#### ✨ Predicted Next Words (Click to append):")
            
            cols = st.columns(len(top_predictions))
            for i, (word, prob) in enumerate(top_predictions):
                with cols[i]:
                    st.button(
                        f"➕ {word} ({prob:.1%})",
                        key=f"btn_{word}_{i}",
                        on_click=append_word,
                        args=(word,),
                        use_container_width=True
                    )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Simple text generation preview
            st.markdown("---")
            st.subheader("🤖 Auto-generated text sample:")
            
            # Generate 10 words sequence preview
            gen_text = text_input
            temp_list = list(token_list)
            
            for _ in range(10):
                padded = pad_sequences([temp_list], maxlen=max_sequence_len-1, padding='pre')
                p = model.predict(padded, verbose=0)[0]
                
                # Use current temperature configuration
                if temperature > 0:
                    p_adj = np.log(p + 1e-10) / temperature
                    exp_p = np.exp(p_adj)
                    p_adj = exp_p / np.sum(exp_p)
                    idx = np.random.choice(len(p_adj), p=p_adj)
                else:
                    idx = np.argmax(p)
                    
                next_w = ""
                for w, index in tokenizer.word_index.items():
                    if index == idx:
                        next_w = w
                        break
                if not next_w:
                    break
                gen_text += " " + next_w
                temp_list.append(idx)
                if len(temp_list) >= max_sequence_len:
                    temp_list.pop(0)
                    
            st.info(f'"{gen_text}..."')
            
    else:
        st.info("💡 Type some words above (e.g. 'to be or not to') to see predictive suggestions.")

if __name__ == '__main__':
    main()
