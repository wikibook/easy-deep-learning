from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import numpy as np

# 더미 데이터
data = np.random.random((1000, 784))
labels = np.random.randint(10, size=(1000, 1))
labels = np_utils.to_categorical(labels, 10)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='rmsprop', 
             loss='categorical_crossentropy', 
             metrics=['accuracy'])

# 학습
model.fit(data, labels, epochs=10)