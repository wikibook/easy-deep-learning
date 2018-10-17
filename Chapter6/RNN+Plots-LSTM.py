# coding: utf-8
# SimpleRNN, LSTM, GRU를 사용한 RNN 예제

# RNN Model 2 - LSTM
# 케라스와 그 외 라이브러리 임포트
from keras.models import Model
from keras.layers import Input, LSTM

# SVG 표시에 필요한 라이브러리 임포트
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# 유닛 수, 스텝 수, 입력 차원, 입력 데이터 형태 정의
units = 10
time_steps = 5
input_dim = 15
input_shape = (time_steps, input_dim)

# 순환 신경망 모델 작성
x = Input(shape=input_shape, name='Input')
y = LSTM(units=units, activation='sigmoid', name='LSTM_1')(x)
model = Model(inputs=[x], outputs=[y])

# SVG 형식으로 모델 표시(그림 6-2-2)
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# 순환 신경망 모델 작성, 시퀀스를 출력
y = LSTM(units=units, activation='sigmoid', return_sequences=True, name='LSTM_1')(x)
model = Model(inputs=[x], outputs=[y])

# SVG 형식으로 모델 표시(그림 6-2-3)
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# 순환 신경망 모델 작성, 내부 상태 표시
x = Input(shape=input_shape, name='Input')
y, state_1, state_2 = LSTM(units=units, activation='sigmoid', return_state=True, name='LSTM_1')(x)
model = Model(inputs=[x], outputs=[y])

# SVG 형식으로 모델 표시(그림 6-2-4)
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))