import pandas as pd
from gensim.models import Word2Vec

def load_data(filepath):
    data = pd.read_csv(filepath, delimiter=';', header=None, names=['sentence','emotion'])
    data = data['sentence']

    gensim_input = []
    for text in data:
        gensim_input.append(text.rstrip().split())
    return gensim_input

input_data = load_data("emotions_train.txt")

# word2vec 모델을 학습하세요.



# happy와 유사한 단어를 확인하세요.
similar_happy = None

print(similar_happy)

# sad와 유사한 단어를 확인하세요.
similar_sad = None
print(similar_sad)

# 단어 good과 bad의 임베딩 벡터 간 유사도를 확인하세요.
similar_good_bad = None

print(similar_good_bad)

# 단어 sad과 lonely의 임베딩 벡터 간 유사도를 확인하세요.
similar_sad_lonely = None

print(similar_sad_lonely)

# happy의 임베딩 벡터를 확인하세요.
wv_happy = None

print(wv_happy)
