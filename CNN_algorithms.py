#####---(Convolutional Neural Network 구조 생성과 학습시키는 부분)---#####
##  설명
#   Deep learning algorithms 중에서 Convolutional Neural Network의 다양한 algorithms을 선택하여
#   구조를 형성하고 학습 및 테스트를 진행하는 함수들이 포함된 library

##  업데이트 기록지
#   2018.03.28.수요일 : 석사 때 짜놓았던 코드를 불러와 좀 더 간편하게 수정 중


#####------------------------------------------------


### 사용할 lib 소환
#   os lib 소환 (system location을 불러오고 지정하기 위해)
import os
#   glob lib 소환 (여러 파일을 한 번에 불러오기 위해)
import glob
#   csv lib 소환 (csv 파일로 불러오기 위해)
import csv
#   numpy lib 소환 (행렬 계산을 위해)
import numpy as np
#   pandas lib 소환 (csv 파일을 불러오기 위해)
import pandas as pd
#   random lib 소환 (data의 순서를 섞기 위해)
import random
#   tqdm에 있는 trange lib 소환 (progress bar를 출력하기 위해)
from tqdm import trange
#   tensorflow lib 소환
import tensorflow as tf


### Weight와 bias를 초기화하는 부분 (초기화하는 방법도 여러가지가 있으니 김성훈 교수님 강의 듣고 업데이트 시키기!!!!!)
##  Weight 초기화
def init_weight(shape, init_type) :
    if init_type == 0 :
        return tf.Variable(tf.random_normal(shape, stddev=0.01))
##  Bias 초기화
def init_bias(shape, init_type) :
    if init_type == 0 :
        return tf.Variable(tf.constant(0.1, shape=shape))




