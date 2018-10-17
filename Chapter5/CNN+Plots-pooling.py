# coding: utf-8
# Conv2D를 사용한 CNN 예제

# CNN Model 6 - Pooling
# 케라스와 그 외 라이브러리를 임포트
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# SVG 표시에 필요한 라이프러리 임포트
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# 윈도우의 경우 다음을 추가
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# 활성화 함수를 ReLU로 지정한 합성곱 신경망 모델 작성
model = Sequential()
model.add(Conv2D(filters=3, kernel_size=(3, 3), input_shape=(6, 6, 1), strides=2, name='Conv2D_1'))
model.add(MaxPooling2D(pool_size=(2, 2), name='MaxPooling2D_1'))

# SVG 형식으로 모델 표시
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))