import re
from sklearn.feature_extraction.text import CountVectorizer

regex = re.compile('[^a-z ]')

with open("text.txt", 'r', encoding='utf-8') as f:
    documents = []
    for line in f:
        # doucments 리스트에 리뷰 데이터를 저장하세요.
        lowered_sent = line.rstrip()
        filtered_sent = regex.sub('', lowered_sent)
        documents.append(filtered_sent)

# CountVectorizer() 객체를 이용해 Bag of words 문서 벡터를 생성하여 변수 X에 저장하세요.
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# 변수 X의 차원을 변수 dim에 저장하세요.
dim = X.shape
# X 변수의 차원을 확인해봅니다.
print(dim)

# 위에서 생성한 CountVectorizer() 객체에서 첫 10개의 칼럼이 의미하는 단어를 words_feature 변수에 저장하세요.
words_feature = vectorizer.get_feature_names_out()[:10]
# CountVectorizer() 객체의 첫 10개 칼럼이 의미하는 단어를 확인해봅니다.
print(words_feature)

# 단어 "comedy"의 빈도수를 idx 변수에 저장하세요.
idx = vectorizer.vocabulary_['comedy']
# 단어 "comedy"의 빈도수를 확인합니다.
print(idx)

# 첫 번째 문서의 Bag of words를 vec1 변수에 저장하세요.
vec1 = X[0]
# 첫 번째 문서의 Bag of words를 확인합니다.
print(vec1)