# ✍️ Shakespearean Next Word Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![LSTM](https://img.shields.io/badge/Model-Bidirectional%20LSTM-purple)

> An end-to-end deep learning project that predicts the **next word** in a sentence, trained on Shakespeare's *Hamlet*. Features a fully interactive Streamlit web app with temperature-controlled creativity, auto-complete, and an in-app training tool.

---

## 🖼️ Demo

| Predictive Keyboard | Auto-Complete Preview |
|---|---|
| Click predicted word buttons to append them live | Generates a 10-word continuation from your input |

> 💡 **Try it yourself** — clone the repo, train the model in one command, and launch the app locally.

---

## 📁 Project Structure

```text
lstm_project/
│
├── app.py                # Streamlit web application
├── train.py              # Model training pipeline
├── requirements.txt      # Python dependencies
├── .gitignore            # Git exclusion rules
├── LICENSE               # MIT License
├── README.md             # Project documentation
│
# ⚠️ Generated at runtime (excluded from Git):
├── model.h5              # Trained LSTM model weights (~19 MB)
├── tokenizer.pickle      # Serialized Keras tokenizer
├── config.json           # Model configuration (vocab size, context window)
└── hamlet.txt            # Raw text corpus (auto-downloaded via NLTK)
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/lstm_project.git
cd lstm_project
```

### 2. Set Up a Virtual Environment *(Recommended)*

```bash
# Create virtual environment
python -m venv .venv

# Activate — Windows:
.venv\Scripts\activate

# Activate — macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python train.py
```

> ⏱️ Takes **~1–2 minutes on CPU**. This automatically downloads Shakespeare's *Hamlet* via NLTK, trains a Bidirectional LSTM for 10 epochs, and saves `model.h5`, `tokenizer.pickle`, and `config.json`.

### 5. Launch the Web App

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## 🎨 App Features

| Feature | Description |
|---|---|
| **Predictive Keyboard** | Dynamically shows the top-k next word predictions with confidence %. Click any word to append it. |
| **Creativity Slider (Temperature)** | Tune generation entropy — low temp = logical, high temp = poetic & creative. |
| **Auto-Complete Preview** | Generates a 10-word continuation of your input text in real time. |
| **In-App Training** | If model files are missing, a one-click **"Start Training Now"** button builds the model directly inside the web app. |

---

## 🧠 Model Architecture

```
Embedding(vocab_size, 128, mask_zero=True)
    → LSTM(150, return_sequences=True, dropout=0.2)
    → LSTM(150, dropout=0.2)
    → Dense(128, activation='relu')
    → Dropout(0.3)
    → Dense(vocab_size, activation='softmax')
```

- **Optimizer**: Adam (lr=0.001, clipnorm=1.0)
- **Loss**: Categorical Cross-Entropy with label smoothing (0.05)
- **Training**: 10 epochs, batch size 64, with EarlyStopping & ReduceLROnPlateau
- **Corpus**: Shakespeare's *Hamlet* (~4,800 unique tokens)

---

## 🚀 Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (only source files — **not** `model.h5` or `tokenizer.pickle`).
2. Log in at [share.streamlit.io](https://share.streamlit.io/).
3. Click **New App** → select your repo, branch `main`, entrypoint `app.py`.
4. Click **Deploy**.
5. Once live, click **"Start Training Now"** inside the app to build the model in the cloud environment.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `tensorflow >= 2.16` | LSTM model (training + inference) |
| `streamlit` | Interactive web application |
| `numpy` | Numerical operations |
| `pandas` | Data handling utilities |
| `scikit-learn` | Train/test split |
| `nltk` | Shakespeare corpus download |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ using TensorFlow & Streamlit</p>
