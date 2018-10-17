# coding: utf-8
# Conv2D를 사용한 CNN 예제

# CNN Model 7 - Flatten and Dense
# 케라스와 그 외 라이브러리 임포트
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.utils import np_utils

# SVG 표시에 필요한 라이브러리 임포트
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# 윈도우의 경우 다음을 추가
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# 합성곱 신경망 출력을 1차원으로 평활화한 후 완전 연결 계층에 전달하는 모델 작성
model = Sequential()
model.add(Conv2D(filters=3, kernel_size=(3, 3), input_shape=(6, 6, 1), padding='same', name='Conv2D_1'))
model.add(Flatten(name='Flatten_1'))
model.add(Dense(units=10, activation='softmax', name='Dense_1'))

# SVG 형식으로 모델 표시
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))