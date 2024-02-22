from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer

files_dir = "./files"

# Função para carregar e pré-processar um Jenkinsfile
def load_and_preprocess_jenkinsfile(file_path):
    with open(file_path, 'r') as file:
        files_content = file.read()
        # Faça qualquer pré-processamento necessário aqui
        # Por exemplo, tokenização, remoção de espaços em branco, etc.
    return files_content

# Lista de caminhos para os arquivos
files_paths = [os.path.join(files_dir, filename) for filename in os.listdir(files_dir)]

# Carregar e pré-processar os Jenkinsfiles
files_data = [load_and_preprocess_jenkinsfile(file_path) for file_path in files_paths]

# Tokenização e padding das sequências
tokenizer = Tokenizer()
tokenizer.fit_on_texts(files_data)
sequences = tokenizer.texts_to_sequences(files_data)

# Determinando o max_length
max_length = max([len(seq) for seq in sequences])

padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Hiperparâmetros
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 16
lstm_units = 64

# Inputs para a pergunta e Jenkinsfile
input_question = Input(shape=(max_length,), dtype=tf.int32, name='input_question')
question_embedding = Embedding(vocab_size, embedding_dim, input_length=max_length)(input_question)

input_file = Input(shape=(max_length,), dtype=tf.int32, name='input_jenkinsfile')
file_embedding = Embedding(vocab_size, embedding_dim, input_length=max_length)(input_file)

# Camada LSTM
lstm_layer = LSTM(lstm_units)

# Processamento das sequências
question_lstm = lstm_layer(question_embedding)
file_lstm = lstm_layer(file_embedding)

# Concatenação das saídas LSTM
concatenated = tf.keras.layers.concatenate([question_lstm, file_lstm])

# Camada densa de saída
output = Dense(vocab_size, activation='softmax')(concatenated)

# Definição do modelo
model = Model(inputs=[input_question, input_file], outputs=output)

# Compilação do modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Preparação dos dados de entrada e saída
x_question = padded_sequences  # Usando os dados já processados para as perguntas
x_file = padded_sequences  # Usando os dados já processados para os files
y = tf.expand_dims(padded_sequences, -1)  # Adicionando uma dimensão extra para cada exemplo

# Treinamento do modelo
model.fit([x_question, x_file], y, epochs=10, batch_size=32)
