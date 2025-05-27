import matplotlib.pyplot as plt
import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
import os
import nltk
torch.manual_seed(404)

with open("test.csv") as csv_f:
    head = "\n".join([next(csv_f) for x in range(5)])
print(head)


# 모델 클래스 정의
class LSTMClassifier(nn.Module):
    # LSTM Classifier 클래스를 정의합니다. Pytorch는 모델을 구성할 때 반드시 nn.Module 클래스를 상속받은 후 이를 토대로 만듭니다.
    def __init__(self, vocab_size, dimension=128):
        # 클래스의 첫 시작인 함수입니다. 여기서 모델에 필요한 여러 변수들을 정의합니다.
        super(LSTMClassifier, self).__init__()

        # LSTM Classifier에 필요한 변수들을 각각 정의합니다.
        self.embedding = nn.Embedding(vocab_size, 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300, hidden_size=dimension, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 3)  # Problem 1

    def forward(self, text, text_len):
        # 모델의 forward feed를 수행하는 함수입니다.
        # text와 text_len 변수를 입력으로 받아 신경망 모델을 forward 방향으로 탈 때 그 출력을 반환합니다.
        # 단어 => encoder => Embedding => 양방향 RNN => Dense => Dense의 구조입니다.
        text_emb = self.embedding(text)

        # 글마다 길이가 다르기에 이를 하나의 batch에서 사용하고자 pack_padded_sequence 함수를 통해 padding을 수행합니다.
        packed_input = pack_padded_sequence(text_emb, text_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        # <ToDo>: model의 마지막에 classification을 위해 dense layer를 추가해주세요.
        text_out = self.fc(text_fea)  # Problem 1

        return text_out


# Test 함수
def test(model = None, device = None, test_loader = None):
    if model == None:
        return 
    # 학습이 끝난 모델을 평가합니다.
    y_pred = list()
    y_true = list()

    # 모델에게 학습이 아닌 평가를 할 것이라고 알립니다.
    model.eval()
    # 학습이 아니기에 최적화를 하지 않는다는 환경을 설정합니다.
    with torch.no_grad():
        # test 데이터를 읽습니다.
        for ((text, text_len), labels), _ in test_loader:
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)

            # model을 함수처럼 호출하면 model에서 정의한 forward 함수가 실행됩니다.
            # 즉, 데이터를 모델에 집어넣어 forward방향으로 흐른 후 그 결과를 받습니다.
            output = model(text, text_len)

            # test는 모델의 최종 결과(class)를 구해야합니다.
            # 모델 출력에서 가장 높은 값을 가지는 index를 구합니다.
            # 그 index가 class 번호가 됩니다.
            _, max_indices = torch.max(output, 1)
            max_indices = max_indices.data.cpu().numpy().tolist()

            y_pred.extend(max_indices)
            y_true.extend(labels.tolist())

    # 모델의 출력과 실제 데이터의 차이를 계산하여 성능을 출력합니다.
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], digits=4))


# 모델 및 기록 저장 불러오기
def save_checkpoint(save_path, model, optimizer, valid_loss):
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)


def load_checkpoint(load_path, model, optimizer, device):
    state_dict = torch.load(load_path, map_location=device)

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return None


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)


def load_metrics(load_path, device):
    state_dict = torch.load(load_path, map_location=device)

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


# 데이터 불러오기


# nltk의 토크나이저를 사용하기에 이를 다운로드 받습니다.
# nltk.download('punkt')

# 데이터의 기본 형태에 대한 정보입니다.
output_file_path="./model/"
os.makedirs(output_file_path, exist_ok=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.long)
text_field = Field(tokenize=word_tokenize, lower=True, include_lengths=True, batch_first=True)
fields = [('text', text_field), ('labels', label_field)]

# train, test 데이터 csv 파일을 읽어옵니다.
train_data, test_data = TabularDataset.splits(path="./", train='train.csv', test='test.csv',
                                  format='CSV', fields=fields, skip_header=True)
train_loader = BucketIterator(train_data, batch_size=32, sort_key=lambda x: len(x.text),
                              device=device, sort=True, sort_within_batch=True)

# <ToDo>: test_dataset을 불러오세요.
test_loader = BucketIterator(test_data, batch_size=32, sort_key=lambda x: len(x.text),
                              device=device, sort=True, sort_within_batch=True) # Problem 2

text_field.build_vocab(train_data, min_freq=3)
vocab_size = len(text_field.vocab)



# 모델 불러오기


model_path = './model/model.pt'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 앞서 정의한 LSTMClassifier 클래스의 인스턴스를 만듭니다.
best_model = LSTMClassifier(vocab_size).to(device)
# Adam optimizier를 정의합니다.
new_optimizer = torch.optim.Adam(best_model.parameters(), lr=0.001)

# <ToDo>: load_checkpoint 함수를 통해 model_path에 있는 모델을 불러옵니다.
load_checkpoint(model_path, best_model, new_optimizer, device)


# 테스트

# <ToDo>: 학습된 모델의 검증을 위해 test 함수의 적절한 parameter를 전달해주세요.
test(best_model, device, test_loader) # Problem 3