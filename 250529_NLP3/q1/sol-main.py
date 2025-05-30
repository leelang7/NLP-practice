from sklearn.model_selection import train_test_split

# 파일을 읽어오세요.
data = []

with open('emotions_train.txt', 'r') as f:
    for line in f:
        line = line.replace("\n", "")
        data.append(line)

# 읽어온 파일을 학습 데이터와 평가 데이터로 분할하세요.
train, test = train_test_split(data, test_size=0.2, random_state=7)

# 학습 데이터셋의 문장과 감정을 분리하세요.
Xtrain = []
Ytrain = []

for sent_train in train:
    sent, emotion = sent_train.split(';')
    Xtrain.append(sent)
    Ytrain.append(emotion)

print(Xtrain)
print(set(Ytrain))

# 평가 데이터셋의 문장과 감정을 분리하세요.
Xtest = []
Ytest = []

for sent_train in test:
    sent, emotion = sent_train.split(';')
    Xtest.append(sent)
    Ytest.append(emotion)

print(Xtest)
print(set(Ytest))