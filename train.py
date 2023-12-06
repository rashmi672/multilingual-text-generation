import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense



tokenizer = Tokenizer()

dataset = "final_english.txt"

# Read the content of the file
with open(dataset, 'r', encoding='utf-8') as file:
    text_content = file.read()

tokenizer.fit_on_texts([text_content])

tokenizer.word_index

input_sequences = []
for sentence in text_content.split('\n'):
  tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

  for i in range(1,len(tokenized_sentence)):
    n_gram = tokenized_sentence[:i+1]
    input_sequences.append(n_gram)
    
    

max_len = max([len(x) for x in input_sequences])


padded_input_sequences = pad_sequences(input_sequences, maxlen = max_len, padding='pre')

X = padded_input_sequences[:,:-1]
y = padded_input_sequences[:,-1]

X.shape
y.shape

from tensorflow.keras.utils import to_categorical
y = to_categorical(y,num_classes=12205)

y.shape

model = Sequential()
model.add(Embedding(12205, 100, input_length=84))
model.add(LSTM(150))
model.add(Dense(12205, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(X,y,epochs=20)
model.save('model_next-word.h5')
