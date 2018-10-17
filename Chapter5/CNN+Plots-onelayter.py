# coding: utf-8
# Conv2D를 사용한 CNN의 예

# CNN Model 1 - one layer
# 케라스와 다른 라이브러리 임포트
from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras.utils import np_utils

# SVG 표시에 필요한 라이브러리 임포트
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# 윈도우의 경우 다음을 추가
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# 합성곱 신경망 모델 작성
model = Sequential()
model.add(Conv2D(filters=3, kernel_size=(3, 3), input_shape=(6, 6, 1), name='Conv2D_1'))

# SVG 형식으로 모델 표시
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))