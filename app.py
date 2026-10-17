
import pickle
import numpy as np
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✍️",
    layout="centered"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #f0f4ff;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #aab4c5;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .prediction-box {
        background: #1b2332;
        border: 1px solid #39465e;
        border-radius: 14px;
        padding: 22px;
        margin-top: 18px;
    }

    .prediction-label {
        color: #aab4c5;
        font-size: 14px;
    }

    .prediction-text {
        color: #8ab4ff;
        font-size: 25px;
        font-weight: 700;
        overflow-wrap: anywhere;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL AND SUPPORTING FILES
# --------------------------------------------------

@st.cache_resource
def load_assets():

    model = load_model(
        "lstm_model.h5",
        compile=False
    )

    # Only load pickle files you trust.
    with open("tokenizer.pickle", "rb") as file:
        tokenizer = pickle.load(file)

    with open("max_len.pickle", "rb") as file:
        max_len = pickle.load(file)

    # Use the model's actual input length when available.
    model_input_len = model.input_shape[1]

    if model_input_len is None:
        model_input_len = int(max_len) - 1

    return model, tokenizer, int(model_input_len)


try:
    model, tokenizer, sequence_length = load_assets()
    assets_loaded = True

except Exception as error:
    assets_loaded = False
    st.error(
        "Unable to load the model or supporting files. "
        "Check that all filenames are correct and that "
        "your TensorFlow environment is compatible."
    )
    st.exception(error)


# --------------------------------------------------
# NEXT WORD PREDICTION FUNCTION
# --------------------------------------------------

def predict_next_word(text):

    # Convert the input text into token IDs.
    token_list = tokenizer.texts_to_sequences([text])[0]

    if not token_list:
        return None

    # Keep the most recent words if the sequence is too long.
    token_list = token_list[-sequence_length:]

    # Pad the sequence to match the model's expected input.
    padded_sequence = pad_sequences(
        [token_list],
        maxlen=sequence_length,
        padding="pre",
        truncating="pre"
    )

    # Predict probabilities for the next token.
    predictions = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # Select the token with the highest probability.
    predicted_index = int(np.argmax(predictions))

    # Convert the predicted token ID back into a word.
    predicted_word = tokenizer.index_word.get(
        predicted_index
    )

    confidence = float(predictions[predicted_index])

    if predicted_word is None:
        return None

    return predicted_word, confidence


# --------------------------------------------------
# STREAMLIT USER INTERFACE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">✍️ Next Word Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An LSTM-powered text prediction experience'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.markdown("### Enter your text")
st.write(
    "Type a sentence below and let your trained LSTM "
    "suggest how it might continue."
)

input_text = st.text_area(
    "Your sentence",
    placeholder="For example: The best way to learn",
    height=130,
    max_chars=2000,
    label_visibility="collapsed"
)

number_of_words = st.slider(
    "Number of words to generate",
    min_value=1,
    max_value=20,
    value=5
)

predict_button = st.button(
    "✨ Predict Next Words",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# DISPLAY PREDICTIONS
# --------------------------------------------------

if predict_button:

    if not assets_loaded:
        st.error("The model could not be loaded.")

    elif not input_text.strip():
        st.warning("Please enter a sentence first.")

    else:
        with st.spinner("Generating your next words..."):

            original_text = input_text.strip()
            generated_text = original_text
            generated_words = []
            confidences = []

            for _ in range(number_of_words):

                result = predict_next_word(generated_text)

                if result is None:
                    break

                next_word, confidence = result

                generated_text += " " + next_word
                generated_words.append(next_word)
                confidences.append(confidence)

        st.divider()
        st.markdown("### 📝 Generated Text")

        st.markdown(
            f'<div class="prediction-box">'
            f'<div class="prediction-label">'
            f'YOUR TEXT CONTINUED'
            f'</div>'
            f'<div class="prediction-text">'
            f'{generated_text}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        if generated_words:
            st.markdown("### 🔮 Predicted Words")

            st.write(" → ".join(generated_words))

            st.caption(
                "The model generates one word at a time, "
                "using each prediction as input for the next."
            )

            with st.expander("View prediction details"):
                for index, (word, confidence) in enumerate(
                    zip(generated_words, confidences),
                    start=1
                ):
                    st.write(
                        f"**Word {index}:** {word} "
                        f"| Model score: {confidence:.2%}"
                    )

            st.download_button(
                label="📥 Download Generated Text",
                data=generated_text,
                file_name="generated_text.txt",
                mime="text/plain",
                use_container_width=True
            )

        else:
            st.warning(
                "No next word could be generated. "
                "Check whether your input contains words "
                "known to the tokenizer."
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Streamlit, TensorFlow and LSTM · "
    "Next Word Predictor"
)
