import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import Adam

# Load the data from the JSON file
with open('../app/data.json', 'r') as file:
    json_data = json.load(file)
    text_data = ' '.join(json_data['data'])  # Combine all sentences into a single string

# Prepare the text data
chars = sorted(list(set(text_data)))  # Get unique characters
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}

# Prepare data for training
SEQ_LENGTH = 40  # Number of characters the model sees before making a prediction
STEP = 3  # Step size for the sliding window

sequences = []
next_chars = []

for i in range(0, len(text_data) - SEQ_LENGTH, STEP):
    sequences.append(text_data[i: i + SEQ_LENGTH])
    next_chars.append(text_data[i + SEQ_LENGTH])

# Vectorize sequences and labels
X = np.zeros((len(sequences), SEQ_LENGTH, len(chars)), dtype=bool)
y = np.zeros((len(sequences), len(chars)), dtype=bool)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Train the model
model.fit(X, y, batch_size=128, epochs=20)

# Text generation function
def generate_text(seed, length):
    generated = seed
    for _ in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, len(chars)))
        for t, char in enumerate(seed):
            if char in char_to_index:
                x_pred[0, t, char_to_index[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = index_to_char[next_index]
        
        generated += next_char
        seed = seed[1:] + next_char  # Move seed window

    return generated

# Generate text using the trained model
seed_text = "Generative AI is "
generated_text = generate_text(seed_text, 100)
print(generated_text)
