# coding: utf-8
# 순환 신경망 예제

# 케라스로 사용할 라이브러리 읽어 들이기
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, LSTM, Dense
from keras.layers.wrappers import TimeDistributed

# IMDB 데이터 세트 읽어 들이기
from keras.datasets import imdb

# SVG 표시에 필요한 라이브러리 임포트
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# 윈도우의 경우 이하 추가
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# 넘파이 등 툴 읽어 들이기
from keras.utils import to_categorical, np_utils
import numpy as np


# 데이터 수, 특징 수, 벡터의 차원 수, 스텝 수 등
train_reviews = 5000
valid_reviews = 100
max_features = 5000
embedding_size = 256
step_size = 5
batch_size = 32
index_from = 2
rnn_units = 128
epochs = 2
word_index_prev = {'<PAD>': 0, '<START>': 1, '<UNK>': 2}

# IMDB 데이터 읽어 들이기
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, index_from=index_from)

# IMDB 데이터로부터 단어 정보 추출
word_index = {word: (index + index_from) for word, index in imdb.get_word_index().items() if (index + index_from) < max_features}
word_index.update(word_index_prev)

# 단어 정보로부터 사전 작성
index_word = {index: word for word, index in word_index.items()}

# 문장을 표시하는 함수
def print_sentence(sentence):
    for index in sentence:
        print(index_word[index], end=" ")
    print()

#첫 문장 표시
print_sentence(x_train[0])

# 학습 데이터와 테스트 데이터 나누기
data_train = [t for s in x_train[:train_reviews] for t in s]
data_valid = [t for s in x_train[train_reviews:train_reviews+valid_reviews] for t in s]

# 배치 처리를 위한 함수 정의
def batch_generator(data, batch_size, step_size):
    seg_len = len(data) // batch_size
    steps_per_epoch = seg_len // step_size
    data_seg_list = np.asarray([data[int(i*seg_len):int((i+1)*seg_len)] for i in range(batch_size)])
    data_seg_list
    i = 0
    while True:
        x = data_seg_list[:, int(i*step_size):int((i+1)*step_size)]
        y = np.asarray([to_categorical(data_seg_list[j, int(i*step_size+1):int((i+1)*step_size+1)], max_features) for j in range(batch_size)])
        yield x, y
        i += 1
        if i >= steps_per_epoch:
            i = 0

# LSTM을 사용한 모델 설계
w = Input(shape=(step_size,), name='Input')
x = Embedding(input_dim=max_features, output_dim=embedding_size, name='Embedding')(w)
y = LSTM(units=rnn_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5, name='LSTM')(x)
w_next = TimeDistributed(Dense(units=max_features, activation='softmax', name='Dense'), name='TimeDistributed')(y)

# 모델 작성
model = Model(inputs=[w], outputs=[w_next])

# 모델 표시
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# 모델 컴파일
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 배치 처리를 위한 데이터 세트 작성
gen_train = batch_generator(data_train, batch_size, step_size)
gen_valid = batch_generator(data_valid, batch_size, step_size)

# 스텝 수 계산
steps_per_epoch_train = len(data_train) / batch_size / step_size
steps_per_epoch_valid = len(data_valid) / batch_size / step_size

# 배치 처리마다 데이터를 학습
model.fit_generator(generator=gen_train, steps_per_epoch=steps_per_epoch_train, epochs=epochs,
                    validation_data=gen_valid, validation_steps=steps_per_epoch_valid)

# 다음에 오는 단어를 선택하는 함수
def sample(preds, temperature=1.0):
    preds = np.log(preds) / temperature
    preds = np.exp(preds) / np.sum(np.exp(preds))
    choices = range(len(preds))
    return np.random.choice(choices, p=preds)

# 임의의 문장을 생성하는 함수
def sample_sentences(num_sentences, sample_sent_len = 20):
    for x_test_i in x_test[:num_sentences]:
        x = np.zeros((1, step_size))
        sentence = x_test_i[:step_size]

        for i in range(sample_sent_len):
            for j, index in enumerate(sentence[-step_size:]):
                x[0, j] = index
            preds = model.predict(x)[0][-1]
            next_index = sample(preds)
            sentence.append(next_index)

        print_sentence(sentence)

# 임의의 문장을 추출
sample_sentences(num_sentences=20, sample_sent_len=15)

# 가중치 정규화
norm_weights = np_utils.normalize(model.get_weights()[0])

# 가까운 의미의 단어를 표시하는 함수
def print_closest_words(word, nb_closest=10):
    index = word_index[word]
    distances = np.dot(norm_weights, norm_weights[index])
    c_indexes = np.argsort(np.squeeze(distances))[-nb_closest:][::-1]
    for c_index in c_indexes:
        print(index_word[c_index], distances[c_index])

# 가까운 의미의 단어
words = ["3",
         "two",
         "great",
         "money",
         "years",
         "look",
         "own",
         "us",
         "using",
        ]

for word in words:
    if word in word_index:
        print('====', word)
        print_closest_words(word)