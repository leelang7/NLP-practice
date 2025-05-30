import nltk

sent_1 = "오늘 중부지방을 중심으로 소나기가 예상됩니다"
sent_2 = "오늘 전국이 맑은 날씨가 예상됩니다"

def cal_jaccard_sim(sent1, sent2):
    # 각 문장을 토큰화 후 set 타입으로 변환하세요.
    words_sent1 = 0
    words_sent2 = 0

    # 공통된 단어의 개수를 intersection 변수에 저장하세요.
    intersection = 0
    
    # 두 문장 내 발생하는 모든 단어의 개수를 union 변수에 저장하세요.
    union = 0

    # intersection과 union을 사용하여 자카드 지수를 계산하고 float 타입으로 반환하세요.
    return 0

# cal_jaccard_sim() 함수 실행 결과를 확인합니다.
print(cal_jaccard_sim(sent_1, sent_2))

# nltk의 jaccard_distance() 함수를 이용해 자카드 유사도를 계산하세요.
nltk_jaccard_sim = 0

# 직접 정의한 함수와 결과가 같은지 비교합니다.
print(nltk_jaccard_sim)