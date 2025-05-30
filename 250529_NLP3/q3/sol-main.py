import pandas as pd
import numpy as np

def cal_partial_freq(texts, emotion):
    filtered_texts = texts[texts['emotion'] == emotion]
    filtered_texts = filtered_texts['sentence']
    partial_freq = dict()

    # 실습 2에서 구현한 부분을 완성하세요.
    for text in filtered_texts:
        words = text.rstrip().split()
        for word in words:
            if word not in partial_freq:
                partial_freq[word] = 1
            else:
                partial_freq[word] += 1

    return partial_freq

def cal_total_freq(partial_freq):
    total = 0
    # 실습 2에서 구현한 부분을 완성하세요.
    for word, freq in partial_freq.items():
        total += freq
    return total

def cal_prior_prob(data, emotion):
    filtered_texts = data[data['emotion'] == emotion]
    # data 내 특정 감정의 로그발생 확률을 반환하는 부분을 구현하세요.
    
    return np.log(len(filtered_texts)/len(data))

def predict_emotion(sent, data):
    emotions = ['anger', 'love', 'sadness', 'fear', 'joy', 'surprise']
    predictions = []
    train_txt = pd.read_csv(data, delimiter=';', header=None, names=['sentence', 'emotion'])

    # 각 감정별 문장 내 단어의 가능도를 계산하세요.
    for emotion in emotions:
        prior_p = cal_prior_prob(train_txt, emotion)
        likelihood_w = 0
        for word in sent.split():
            e_freq = cal_partial_freq(train_txt, emotion)
            likelihood_w += np.log((e_freq[word] + 10)/(cal_total_freq(e_freq) + 10))

        predictions.append((emotion, prior_p + likelihood_w))
        predictions.sort(key=lambda x: x[1])

    return predictions[-1]

# 아래 문장의 예측된 감정을 확인해보세요.
test_sent = "i really want to go and enjoy this party"
print(predict_emotion(test_sent, "emotions_train.txt"))