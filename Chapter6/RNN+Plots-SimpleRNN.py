# coding: utf-8
# SimpleRNN, LSTM, GRU를 사용한 RNN 예제

# RNN Model 1 - SimpleRNN
# 케라스와 그 외 라이브러리 임포트
from keras.models import Model
from keras.layers import Input, SimpleRNN

# SVG 표시에 필요한 라이브러리 임포트
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# 윈도우의 경우 이하 추가
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# 유닛 수, 스텝 수, 입력 차원 수, 입력 데이터의 형태를 정의
units = 10
time_steps = 5
input_dim = 15
input_shape = (time_steps, input_dim)

# 순환 신경망 모델 작성
x = Input(shape=input_shape, name='Input')
y = SimpleRNN(units=units, activation='sigmoid', name='SimpleRNN_1')(x)
model = Model(inputs=[x], outputs=[y])

# SVG 형식으로 표시(그림 6-1-5)
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# 순환 신경망 모델 작성, 시퀀스 출력
y = SimpleRNN(units=units, activation='sigmoid', return_sequences=True, name='SimpleRNN_1')(x)
model = Model(inputs=[x], outputs=[y])

# SVG 형식으로 모델 표시(그림 6-1-6)
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# 순환 신경망 모델 작성, 내부 상태도 출력
y, state = SimpleRNN(units=units, activation='sigmoid', return_state=True, name='SimpleRNN_1')(x)
model = Model(inputs=[x], outputs=[y])

# SVG 형식으로 모델 표시(그림 6-1-7)
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))