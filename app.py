# flask_app/app.py
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


app = Flask(__name__)

# Load the model and resources during initialization
english_model = None
english_tokenizer = None
hindi_model = None
hindi_tokenizer = None

@app.before_first_request
def load_model_and_resources():
    global english_model, english_tokenizer, hindi_model, hindi_tokenizer
    english_model = load_model("English/model_next-word.h5")
    english_tokenizer = convert_to_tokenizer("English/tokenizer-english.pickle")
    hindi_model = load_model("Hindi/model_hin_nxt.h5")
    hindi_tokenizer = convert_to_tokenizer('Hindi/tokenizer.pickle')

def convert_to_tokenizer(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_sequence():
    prompt = request.form.get('prompt')
    no_words = int(request.form.get('no_words'))
    language = request.form['language']
    if language == "English":
        sentence = generate_seq_eng(prompt, no_words, english_tokenizer, english_model)
        return jsonify({'generated_sequence': sentence, 'no_words': no_words})
    
    elif language == "Hindi":
        sentence = generate_seq_hindi(hindi_model, hindi_tokenizer, 2, prompt, no_words)
        return jsonify({'generated_sequence': sentence, 'no_words': no_words})
    
    return render_template('index.html')

    # sentence = prompt
    # text_collection = prompt.split(" ")
    # word = text_collection[-1]
    # for i in range(no_words):
    #     next = next_word([word])
    #     sentence = sentence + " " + next
    #     word = next
    # return jsonify({'generated_sequence': sentence, 'no_words': no_words})

def generate_seq_hindi(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        # predict probabilities for each word
        yhat = model.predict(encoded, verbose=0)
        # get the index of the word with the highest probability
        predicted_word_index = np.argmax(yhat)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    return in_text

def generate_seq_eng(text, no_words, tokenizer, model):
    for i in range(no_words):
        # tokenize
        token_text = tokenizer.texts_to_sequences([text])[0]
        # padding
        padded_token_text = pad_sequences([token_text], maxlen=84, padding='pre')
        # predict
        pos = np.argmax(model.predict(padded_token_text))

        for word,index in tokenizer.word_index.items():
            # print("hi")
            if index == pos:
                text = text + " " + word
    return text

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, threaded=True)