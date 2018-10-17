# coding: utf-8
# CNN 을 사용한 학습 예제

# CNN Train 1
# 케라스와 그 외 라이브러리 임포트
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.utils import np_utils
from keras.optimizers import SGD

# 조기 종료를 위한 Callbacks과 데이터 처리에 필요한 Numpy 임포트
import keras.callbacks as callbacks
import numpy as np

# SVG 표시에 필요한 라이브러리 임포트
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# 윈도우의 경우 다음을 추가
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# 임의의 숫자로 더미 데이터를 준비
x_train = np.random.random((100, 6, 6, 1))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 6, 6, 1))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

# 합성곱 신경망 모델 작성
model = Sequential()
model.add(Conv2D(filters=3, kernel_size=(3, 3), input_shape=(6, 6, 1), kernel_initializer='lecun_uniform', name='Conv2D_1'))
model.add(Flatten(name='Flatten_1'))
model.add(Dense(units=10, activation='softmax', name='Dense_1'))

# 시퀀스 출력
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# 조기 종료 설정
earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=5)

# 모델 컴파일
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(x_train, y_train, batch_size=32, epochs=10, callbacks=[earlyStopping], validation_split=0.2)