import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

raw_text = pd.read_csv("emotions_train.txt", delimiter=';', header=None, names=['sentence','emotion'])
train_data = raw_text['sentence']
train_emotion = raw_text['emotion']

# CountVectorizer 객체인 변수 cv를 만들고, fit_transform 메소드로 train_data를 변환하세요.
cv = None
transformed_text = None

# MultinomialNB 객체인 변수 clf를 만들고, fit 메소드로 지시사항 1번에서 변환된 train_data와 train_emotion을 학습하세요.
clf = None

# 아래 문장의 감정을 예측하세요.
test_data = ['i am curious', 'i feel gloomy and tired', 'i feel more creative', 'i feel a little mellow today']

test_result = None
print(test_result)