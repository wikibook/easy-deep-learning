# coding: utf-8
# Matplotlib example

# 필요한 라이브러리 읽어 들이기
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# 출력 데이터 준비
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

# 그래프 그리기
fig, ax = plt.subplots()
ax.plot(t, s)

# 그래프의 레이블과 그리드 그리기
ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

# 그래프 표시
plt.show()