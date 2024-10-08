import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data from JSON
with open('app/data.json') as f:
    data = json.load(f)

ideas = data['ideas']

# Extract titles and descriptions for vocab
vocab = sorted(set(' '.join(idea['title'] + " " + idea['description'] for idea in ideas).split()))
word_to_index = {word: idx for idx, word in enumerate(vocab)}  # Start from 0
index_to_word = {idx: word for word, idx in word_to_index.items()}

# Prepare training data
X, y = [], []
for idea in ideas:
    # Combine title and description for better context
    text = idea['title'] + " " + idea['description']
    tokenized = [word_to_index[word] for word in text.split()]
    for i in range(1, len(tokenized)):
        X.append(tokenized[:i])  # Use words before the current word as input
        y.append(tokenized[i])    # Current word as target

# Pad sequences to ensure uniform input size
max_length = max(len(seq) for seq in X)
X = pad_sequences(X, maxlen=max_length, padding='pre')
y = np.array(y)

# Define the neural network model
model = Sequential()
model.add(Embedding(input_dim=len(vocab), output_dim=50))  # Increased output_dim for better representation
model.add(LSTM(100, return_sequences=True))  # Return sequences for better context
model.add(LSTM(50))  # Additional LSTM layer
model.add(Dense(len(vocab), activation='softmax'))  # Ensure output matches vocabulary size

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50)  # Increase epochs

# Function to sample predictions
def sample(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

# Function to generate ideas
def generate_idea(prompt, max_words=20, temperature=1.0):
    tokenized_prompt = [word_to_index[word] for word in prompt.split() if word in word_to_index]
    tokenized_prompt = pad_sequences([tokenized_prompt], maxlen=max_length, padding='pre')
    
    generated_idea = []
    
    for _ in range(max_words):
        prediction = model.predict(tokenized_prompt)[0]
        generated_word_index = sample(prediction, temperature)
        generated_word = index_to_word[generated_word_index]
        
        # Check if generated word is a valid token
        if generated_word not in index_to_word.values():  # Avoids empty tokens
            continue
        
        generated_idea.append(generated_word)
        
        # Update the prompt with the generated word
        tokenized_prompt = pad_sequences([tokenized_prompt[0].tolist() + [generated_word_index]], maxlen=max_length, padding='pre')

    return ' '.join(generated_idea)
