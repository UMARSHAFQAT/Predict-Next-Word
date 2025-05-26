import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model('next_word_lstm.h5')

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Predict function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    predicted = model.predict(token_list, verbose=0)
    predicted_index = int(np.argmax(predicted, axis=1)[0])
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None

# Streamlit UI
st.title("Predict Next Word")
input_text = st.text_input("Enter a sequence of words:")

if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    if next_word:
        st.write(f"**Next Word:** {next_word}")
    else:
        st.write("⚠️ Could not predict the next word. Try a different input.")
