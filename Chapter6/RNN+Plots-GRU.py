# coding: utf-8
# SimpleRNN, LSTM, GRU를 사용한 RNN 예제

# RNN Model 3 - GRU
# 케라스와 그 외 라이브러리 임포트
from keras.models import Model
from keras.layers import Input, GRU

# SVG 표시에 필요한 라이브러리 임포트
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# 유닛 수, 스텝 수, 입력 차원 수, 입력 데이터의 형태를 정의
units = 10
time_steps = 5
input_dim = 15
input_shape = (time_steps, input_dim)

# 순환 신경망 모델 작성
x = Input(shape=input_shape, name='Input')
y = GRU(units=units, activation='sigmoid', name='GRU_1')(x)
model = Model(inputs=[x], outputs=[y])

# SVG 형식으로 모델 표시(그림 6-3-2)
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# 순환 신경망 모델 작성, 시퀀스 출력
y = GRU(units=units, activation='sigmoid', return_sequences=True, name='GRU_1')(x)
model = Model(inputs=[x], outputs=[y])

# SVG 형식으로 모델 표시(그림 6-3-3)
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# 순환 신경망 모델 작성, 내부 상태 출력
y, state = GRU(units=units, activation='sigmoid', return_state=True, name='GRU_1')(x)
model = Model(inputs=[x], outputs=[y])

# SVG 형식으로 모델 표시(그림 6-3-4)
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))