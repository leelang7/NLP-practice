data = ['this is a dog', 'this is a cat', 'this is my horse','my name is elice', 'my name is hank']

def count_unigram(docs):
    unigram_counter = dict()
    # docs에서 발생하는 모든 unigram의 빈도수를 딕셔너리 unigram_counter에 저장하여 반환하세요.
    
    return unigram_counter

def count_bigram(docs):
    bigram_counter = dict()
  # docs에서 발생하는 모든 bigram의 빈도수를 딕셔너리 bigram_counter에 저장하여 반환하세요.
    
    return bigram_counter

def cal_prob(sent, unigram_counter, bigram_counter):
    words = sent.split()
    result = 1.0
    # sent의 발생 확률을 계산하여 변수 result에 저장 후 반환하세요.
    
    return result

# 주어진data를 이용해 unigram 빈도수, bigram 빈도수를 구하고 "this is elice" 문장의 발생 확률을 계산해봅니다.
unigram_counter = count_unigram(data)
bigram_counter = count_bigram(data)
print(cal_prob("this is elice", unigram_counter, bigram_counter))
