
# Next Word Predictor ✍️

An LSTM-based next word prediction application built using
TensorFlow and Streamlit.

Deployed Link - https://nextwordpredictor-jdqr2h9afleqybqxeyuav3.streamlit.app/

## Features

- Predicts the next word from user-provided text
- Generates multiple words iteratively
- Displays model prediction scores
- Allows users to download generated text

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Streamlit
- LSTM (Long Short-Term Memory)

## Project Structure

```text
Next_word_predictor/
├── app.py
├── lstm_model.h5
├── tokenizer.pickle
├── max_len.pickle
├── qoute_dataset.csv
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Next_word_predictor.git
cd Next_word_predictor
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install streamlit tensorflow numpy
```

## Run the App

```bash
streamlit run app.py
```

## How It Works

The trained LSTM model predicts the next word from a
tokenized input sequence. Repeated predictions allow the
app to generate a longer text continuation.

## Author

Mohammad Zeeshan
