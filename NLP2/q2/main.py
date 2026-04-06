import re

word_counter = dict()
regex = re.compile('[^a-zA-Z]')

# 텍스트 파일을 소문자로 변환 및 숫자 및 특수기호를 제거한 딕셔너리를 만드세요.
with open('text.txt', 'r') as f: # 실습 1 과 동일한 방식으로 `IMDB dataset`을 불러옵니다.
    for line in f:
        pass



# 단어 "the"의 빈도수를 확인해 보세요.
count = 0

print(count)