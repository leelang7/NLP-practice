# 경고문을 무시합니다.
import warnings
warnings.filterwarnings(action='ignore')

import pickle
import tensorflow as tf
import numpy as np

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
        ])
    return model

def generate_text(model, start_string):
    num_generate = 100

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) 
        predicted_id = np.argmax(predictions[-1])
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

model = build_model(65, 256, 1024, batch_size=1)

# model.load_weights()을 이용해 데이터를 불러오세요.
model.load_weights(tf.train.latest_checkpoint("checkpoints/"))
model.build()

with open('word_index.pkl', 'rb') as f:
    char2idx, idx2char = pickle.load(f)

# model.summary()

# "Juliet: "이라는 문자열을 추가하여 생성된 문장을 result 변수에 저장하세요.
result = generate_text(model, start_string="Juliet: ")
print(result)