from gensim.models.fasttext import FastText
import numpy as np


def compute_similarity(model, word1, word2):
    """
    두 단어의 유사도를 계산하는 함수
    :param model: fastText model
    :param word1: 첫 번째 단어
    :param word2: 두 번째 단어
    :return: 두 단어의 코사인 유사도
    """
    try:
        similarity = model.wv.similarity(word1, word2)
    except KeyError as e:
        # 사전에 없는 단어 예외 처리
        print(f"단어를 찾을 수 없습니다: {e}")
        similarity = None

    return similarity

def get_word_by_calculation(model, word1, word2, word3):
    """
    단어 벡터의 연산 결과로 가장 적합한 단어를 찾는 함수
    :param model: fastText model
    :param word1: 기준 단어
    :param word2: 뺄 단어
    :param word3: 더할 단어
    :return: 벡터 연산 결과의 가장 가까운 단어
    """
    try:
        output_word = model.wv.most_similar(positive=[word1, word3], negative=[word2], topn=1)[0][0]
    except KeyError as e:
        print(f"단어를 찾을 수 없습니다: {e}")
        output_word = None

    return output_word

def get_similar_word_from_oov(model, word1):
    """
    OOV 단어에 대한 유사 단어를 찾는 함수
    :param model: fastText model
    :param word1: 입력 단어
    :return: 사전에 없는 단어일 경우 유사한 단어, 있을 경우 원래 단어
    """
    try:
        if word1 in model.wv:
            return word1
        else:
            similar_word = model.wv.most_similar(word1, topn=1)[0][0]
            return similar_word
    except KeyError as e:
        print(f"단어를 찾을 수 없습니다: {e}")
        return None

def main():
    # FastText 모델 경로 설정
    model_path = './data/fasttext_model'
    
    try:
        # 모델을 로드합니다.
        model = FastText.load(model_path)
    except Exception as e:
        print(f"모델을 로드할 수 없습니다: {e}")
        return
    
    # 두 단어의 유사도를 찾습니다.
    word1 = "이순신"
    word2 = "원균"
    word1_word2_sim = compute_similarity(model, word1, word2)
    print("{}와/과 {} 유사도: {}".format(word1, word2, word1_word2_sim))

    
    # 단어 벡터 연산
    word1 = "대통령"
    word2 = "현대"
    word3 = "고대"
    cal_result = get_word_by_calculation(model, word1, word2, word3)
    print("{} - {} + {}: {}".format(word1, word2, word3, cal_result))
    
    # '컴퓨터' 단어의 유사성 검사
    oov_word = "컴퓨터"
    oov_word_result = get_similar_word_from_oov(model, oov_word)
    if oov_word == oov_word_result:
        print("단어 '{}'는 fastText가 알고 있음".format(oov_word))
    else:
        print("{}와 근접한 단어: {}".format(oov_word, oov_word_result))
    
    # 오타 단어 '캄퓨터'의 유사 단어 찾기
    oov_word = "캄퓨터"
    oov_word_result = get_similar_word_from_oov(model, oov_word)
    if oov_word == oov_word_result:
        print("단어 '{}'는 fastText가 알고 있음".format(oov_word))
    else:
        print("{}와 근접한 단어: {}".format(oov_word, oov_word_result))

    return word1_word2_sim, cal_result, oov_word_result


if __name__ == "__main__":
    main()
