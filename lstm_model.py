import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # For sentiment labeling


def load_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(subset=['Headline'], inplace=True)
    return df['Headline'].values


def generate_sentiment_scores(headlines):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(text)['compound'] for text in headlines]
    return np.array(scores)  


def preprocess_data(headlines, max_words=5000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(headlines)
    sequences = tokenizer.texts_to_sequences(headlines)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences, tokenizer


def load_glove_embeddings(glove_path, tokenizer, embedding_dim=300):
    embeddings_index = {}
    
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    
    with open(glove_path, 'r', encoding='utf-8') as f:
        
        for line in f:
            values = line.strip().split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except ValueError:
                print(f"Skipping line: {line}")
                
            for word, i in tokenizer.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector 
                
    return embedding_matrix


def build_lstm_model(max_words, max_len, embedding_matrix, embedding_dim=300):
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1,  
                  output_dim=embedding_dim,  
                  weights=[embedding_matrix],  
                  trainable=False,  
                  input_length=max_len),  
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='linear')  
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


if __name__ == "__main__":
    
    glove_path = "D:/Qwegle/Qwegle/Algorithmic_Trading/glove.840B.300d.txt"
    sentiment_data_path = "D:/Qwegle/Qwegle/Algorithmic_Trading/sentiment_data.csv"
    
    max_words = 5000
    max_len = 100
    embedding_dim = 300
    batch_size = 32
    epochs = 10
    
    headlines = load_data(sentiment_data_path)
    x, tokenizer = preprocess_data(headlines, max_words, max_len)
    
    y = generate_sentiment_scores(headlines)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42)
    embedding_matrix = load_glove_embeddings(
        glove_path, tokenizer, embedding_dim)
    model = build_lstm_model(
        max_words, max_len, embedding_matrix, embedding_dim)
    checkpoint = ModelCheckpoint(
        "lstm.h5", save_best_only=True, monitor='val_loss', mode='min')
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint]
    )

    with open('tokenizer1.pkl', 'wb') as file:
        pickle.dump(tokenizer, file)
    
    print("Training completed. Model saved as lstm.h5.")
    print("Tokenizer saved as tokenizer1.pkl.")
