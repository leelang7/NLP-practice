from gensim.models import FastText
import pandas as pd

# Emotions dataset for NLP 데이터셋을 불러오는 load_data() 함수입니다.
def load_data(filepath):
    data = pd.read_csv(filepath, delimiter=';', header=None, names=['sentence','emotion'])
    data = data['sentence']

    gensim_input = []
    for text in data:
        gensim_input.append(text.rstrip().split())

    return gensim_input

input_data = load_data("emotions_train.txt")

# fastText 모델을 학습하세요.


# day와 유사한 단어 10개를 확인하세요.
similar_day = None

print(similar_day)

# night와 유사한 단어 10개를 확인하세요.
similar_night = None

print(similar_night)

# elllllllice의 임베딩 벡터를 확인하세요.
wv_elice = None

print(wv_elice)