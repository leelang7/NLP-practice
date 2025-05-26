import matplotlib.pyplot as plt
import torch
from torchtext.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
import os
import nltk

torch.manual_seed(404)

with open("train.csv") as csv_f:
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

        self.fc = nn.Linear(2*dimension, 3)

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
        text_out = None # Problem 1

        return text_out


# Train 함수


def train(model, device, optimizer, train_loader, valid_loader, output_file_path, num_epochs):
    if valid_loader == None:
        return
    # 학습에 필요한 변수들을 기본적으로 정의합니다.
    running_loss = 0.0
    global_step = 0
    train_loss_list = list()
    valid_loss_list = list()
    global_steps_list = list()
    loss_fn = nn.CrossEntropyLoss()
    best_valid_loss = float("Inf")
    eval_every = 10

    # model에게 학습이 진행됨을 알려줍니다.
    model.train()
    # num_epochs만큼 epoch을 반복합니다.
    for epoch in range(num_epochs):
        # train_loader를 읽으면 정해진 데이터를 읽어옵니다.
        for ((text, text_len), labels), _ in train_loader:
            # 데이터를 GPU로 옮깁니다.
            text = text.to(device)
            text_len = text_len.to(device)
            labels = labels.to(device)
            
            # model을 함수처럼 호출하면 model에서 정의한 forward 함수가 실행됩니다.
            # 즉, 데이터를 모델에 집어넣어 forward방향으로 흐른 후 그 결과를 받습니다.
            output = model(text, text_len)

            # forward 결과와 실제 데이터 결과의 차이를 정의한 loss 함수로 구합니다.
            loss = loss_fn(output, labels)

            # 최적화 수행
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % eval_every == 0:
                # 100번에 한 번으로 validation 데이터를 이용하여 성능을 검증합니다.
                average_train_loss, average_valid_loss = evaluate(model, device, valid_loader, loss_fn,
                                                                  running_loss, eval_every)
                
                # 검증이 끝난 후 다시 모델에게 학습을 준비시킵니다.
                running_loss = 0.0
                model.train()

                # 결과 출력
                print('Epoch {}, Step {}, Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, global_step, average_train_loss, average_valid_loss))

                # 결과 저장
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # 만약 기존 것보다 성능이 높게 나왔다면 현재 모델 상태를 저장합니다.
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(output_file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(output_file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    # 결과를 저장합니다.
    save_metrics(output_file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)


# evaluate 함수


def evaluate(model, device, valid_loader, loss_fn, running_loss, eval_every):
    # 학습 중 모델을 평가합니다.
    # 모델에게 학습이 아닌 평가를 할 것이라고 알립니다.
    model.eval()
    valid_running_loss = 0.

    # 학습이 아니기에 최적화를 하지 않는다는 환경을 설정합니다.
    with torch.no_grad():
        # validation 데이터를 읽습니다.
        for ((text, text_len), labels), _ in valid_loader:
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            
            # model을 함수처럼 호출하면 model에서 정의한 forward 함수가 실행됩니다.
            # 즉, 데이터를 모델에 집어넣어 forward방향으로 흐른 후 그 결과를 받습니다.
            output = model(text, text_len)

            # validation 데이터의 loss, 즉 모델의 출력과 실제 데이터의 차이를 구합니다.
            loss = loss_fn(output, labels)
            valid_running_loss += loss.item()

    # 평균 loss를 계산합니다.
    average_train_loss = running_loss / eval_every
    average_valid_loss = valid_running_loss / len(valid_loader)

    return average_train_loss, average_valid_loss


# 그래프 그리는 함수


def draw_graph(output_file_path, device):
    try:
        train_loss_list, valid_loss_list, global_steps_list = load_metrics(output_file_path + '/metrics.pt', device)
        plt.plot(global_steps_list, train_loss_list, label='Train')
        plt.plot(global_steps_list, valid_loss_list, label='Valid')
        plt.xlabel('Global Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("train_valid_loss.png", bbox_inches='tight')
        elice_utils.send_image("train_valid_loss.png")
    except:
        return


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

    return state_dict['valid_loss']


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

# train, validation 데이터 csv 파일을 읽어옵니다.
train_data, valid_data = TabularDataset.splits(path="./", train='train.csv', validation='valid.csv',
                                               format='CSV', fields=fields, skip_header=True)
train_loader = BucketIterator(train_data, batch_size=32, sort_key=lambda x: len(x.text),
                              device=device, sort=True, sort_within_batch=True)

# <ToDo>: valid_dataset을 불러오세요.
valid_loader = None # Problem 2

text_field.build_vocab(train_data, min_freq=3)
vocab_size = len(text_field.vocab)



# 모델 학습


# 앞서 정의한 LSTMClassifier 클래스의 인스턴스를 만듭니다.
model = LSTMClassifier(vocab_size).to(device)
# Adam optimizier를 사용합니다.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train함수를 이용해 학습합니다.
train(model, device, optimizer, train_loader, valid_loader, output_file_path, 1)


# 결과 출력 

draw_graph(output_file_path, device)