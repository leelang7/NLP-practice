import pandas as pd
import numpy as np

def cal_partial_freq(texts, emotion):
    filtered_texts = texts[texts['emotion'] == emotion]
    filtered_texts = filtered_texts['sentence']
    partial_freq = dict()

    for text in filtered_texts:
        # 전체 데이터 내 각 단어별 빈도수를 입력해 주는 부분을 구현하세요.
        words = text.rstrip().split()
        for word in words:
            if word not in partial_freq:
                partial_freq[word] = 1
            else:
                partial_freq[word] += 1

    return partial_freq

def cal_total_freq(partial_freq):
    total = 0
    # partial_freq 딕셔너리에서 감정별로 문서 내 전체 단어의 빈도수를 계산하여 반환하는 부분을 구현하세요.
    for word, freq in partial_freq.items():
        total += freq
    return total

data = pd.read_csv("emotions_train.txt", delimiter=';', header=None, names=['sentence', 'emotion'])

# happy가 joy라는 감정을 표현하는 문장에서 발생할 가능도를 구하세요.
joy_freq = cal_partial_freq(data, 'joy')
joy_likelihood = joy_freq['happy']/cal_total_freq(joy_freq)
print(joy_likelihood)

# can이 surprise라는 감정을 표현하는 문장에서 발생할 가능도를 구하세요.
sad_freq = cal_partial_freq(data, 'sadness')
sad_likelihood = sad_freq['happy']/cal_total_freq(sad_freq)
print(sad_likelihood)

# can이 surprise라는 감정을 표현하는 문장에서 발생할 가능도를 구하세요.
sup_freq = cal_partial_freq(data, 'surprise')
sup_likelihood = sup_freq['can']/cal_total_freq(sup_freq)
print(sup_likelihood)