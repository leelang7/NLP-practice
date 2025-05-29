import pandas as pd

def cal_partial_freq(texts, emotion):
    partial_freq = dict()
    filtered_texts = texts[texts['emotion']==emotion]
    filtered_texts = filtered_texts['sentence']
    
    # 전체 데이터 내 각 단어별 빈도수를 입력해 주는 부분을 구현하세요.
    
    
    return partial_freq

def cal_total_freq(partial_freq):
    total = 0
    # partial_freq 딕셔너리에서 감정별로 문서 내 전체 단어의 빈도수를 계산하여 반환하는 부분을 구현하세요.
    
    return total

# Emotions dataset for NLP를 불러옵니다.
data = pd.read_csv("emotions_train.txt", delimiter=';', header=None, names=['sentence','emotion'])

# happy가 joy라는 감정을 표현하는 문장에서 발생할 가능도를 구하세요.
joy_likelihood = None
print(joy_likelihood)

# happy가 sadness라는 감정을 표현하는 문장에서 발생할 가능도를 구하세요.
sad_likelihood = None
print(sad_likelihood)

# can이 surprise라는 감정을 표현하는 문장에서 발생할 가능도를 구하세요.
sup_likelihood = None
print(sup_likelihood)
