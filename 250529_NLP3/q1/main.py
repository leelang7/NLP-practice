from sklearn.model_selection import train_test_split

# 파일을 읽어오세요.
data = []
with open('emotions_train.txt', 'r') as f:
    pass


# 읽어온 파일을 학습 데이터와 평가 데이터로 분할하세요.
train, test = None, None


# 학습 데이터셋의 문장과 감정을 분리하세요.
Xtrain = []
Ytrain = []

print(Xtrain)
print(set(Ytrain))

# 평가 데이터셋의 문장과 감정을 분리하세요.
Xtest = []
Ytest = []

print(Xtest)
print(set(Ytest))