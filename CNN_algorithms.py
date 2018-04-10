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
#   math lib 소환 (올림하기 위해)
import math
import matplotlib.pyplot as plt
#####------------------------------------------------



#####------------------------------------------------
### Weight와 bias를 초기화하는 부분 (초기화하는 방법도 여러가지가 있으니 김성훈 교수님 강의 듣고 업데이트 시키기!!!!!)
##  Weight 초기화
def init_weight(shape, init_type) :

    '''

    Input
        shape       :   만들려는 weight의 shape
        init_type   :   초기값을 어떠한 유형으로 설정할 것인지
                        (0 : 설정한 표준편차에 의거하여 랜덤하게 값을 지정하는 방법, 1 : )

    Output
        원하는 크기에 원하는 유형으로 초기화된 weight

    '''


    if init_type == 0 :
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

##  Bias 초기화
def init_bias(shape, init_type) :

    '''

    Input
        shape       :   만들려는 bias의 shape
        init_type   :   초기값을 어떠한 유형으로 설정할 것인지
                        (0 : 설정한 값으로 지정하는 방법, 1 : )

    Output
        원하는 크기에 원하는 유형으로 초기화된 bias

    '''


    if init_type == 0 :
        return tf.Variable(tf.constant(0.1, shape=shape))



### 모든 layer에 필요한 weight와 bias를 설정한 모양으로 만들기 위한 함수
def make_wei_bias (conv_deep, conv_width, kernel_size, kernel_num, pool_num, pool_size, pool_str,
                   fc_layers_num, fc_layers_units,
                   input_data_row, input_data_col, input_data_depth, class_num,
                   weight_init_type, bias_init_type) :

    '''

    Input
        conv_deep           :   얼마나 많은 convolutional layer를 만들지 지정 ex) 3 : conv layer를 3개 만듬
        conv_width          :   Convolutional layer들의 각 layer마다 kernel의 종류가 얼마나 되는지에 대한 정보가 담긴 리스트 (각 원소가 차례대로 각 conv layer의 kernel 종류의 수를 의미)
        kernel_size         :   Convolutional layer들에 들어갈 모든 kernel의 크기를 입력한 행렬 (행 : 각 layer에 있는 모든 kernel들의 크기 / 열 : 각 layer의 각 kernel의 크기)
        kernel_num          :   Convolutional layer들에 들어갈 모든 kernel의 개수를 입력한 행렬 (행 : 각 layer에 있는 모든 kernel들의 개수 / 열 : 각 layer의 각 kernel의 개수)
        pool_num            :   Pooling layer의 개수
        pool_size           :   Pooling layer의 filter size (모든 pooling layer의 filter size를 동일하게 설정)
        pool_str            :   Pooling layer의 stride 값 (모든 pooling layer의 stride를 동일하게 설정)

        fc_layers_num       :   Fully-connected layers의 개수
        fc_layers_units     :   모든 fully-connected layer들의 unit 개수들에 대한 정보가 담긴 리스트 (각 원소가 차례대로 각 fc layer의 units 수)

        input_data_row      :   Input data에서 한 data의 행렬에서 row의 크기
        input_data_col      :   Input data에서 한 data의 행렬에서 column의 크기
        input_data_depth    :   Input data의 depth로 맨 처음 conv layer가 input으로 받는 depth를 설정해주기 위해 필요
        class_num           :   Class의 개수

        weight_init_type    :   Weight의 초기값을 어떠한 유형으로 설정할 것인지
        bias_init_type      :   Bias의 초기값을 어떠한 유형으로 설정할 것인지

    Output
        weight_names_list   :   모든 weight(or bias)의 이름들이 저장된 리스트
        weights_dic         :   모든 weight의 값이 weight 이름을 key로 가지며 저장된 딕셔너리
        biases_dic          :   모든 bias의 값이 weight 이름을 key로 가지며 저장된 딕셔너리
        fc_input_length     :   첫 fully-connected layer의 input의 길이 (1xN 형태로 쭉 펴졌을 때의 N의 값)

    '''


    ### 설정한 크기와 유형에 의거하여 초기화된 모든 weight와 bias를 저장할 변수들과
    ### fully-connected layer로 들어갈 때 일렬로 펴주는 과정에서 직전 layer의 ouptut depth(merge한 결과의 depth)를 고려해야하기 때문에 해당 depth 크기를 저장할 변수 초기화
    #   Weight와 bias의 이름들을 저장할 리스트 초기화
    #   Ex) ['conv1_1', 'conv1-2', ... , 'fc_1', ..., 'out']
    weight_names_list = []
    #   각 layer(conv에서는 각 kernel까지)에 존재하는 weight들을 저장할 딕셔너리 초기화
    #   Ex) {'conv1_1' : [[0.001, 0.002, ... , 0.003], ... ]. ... }
    weights_dic = {}
    #   각 layer(conv에서는 각 kernel까지)에 존재하는 bias들을 저장할 딕셔너리 초기화
    #   Ex) {'conv1_1' : [[0.001], [0.002], ... ], ... }
    biases_dic = {}
    #   첫 fully-connected layer로 들어갈 때 일렬로 펴주는 과정에서 직전 layer의 output depth(merge한 결과의 depth)를 파악해야 하기 때문에 해당 depth의 값을 저장할 변수 정수 초기화
    #inal_depth = 0


    ### Convolutional layer에서 원하는 모양의 kernel을 만드는 부분 (kernel 자체에 weight와 bias가 내포되어 있도록)
    ##  한 convolutional layer부터 차근차근 생성
    print('Convolutional layer에 해당하는 weight와 bias를 생성하고 있습니다.')
    for i in trange(conv_deep) :

        ##  입력한 모든 layer의 넓이(width), kernel의 크기, kernel의 개수 중에서 해당 layer에 대한 정보가 담긴 행을 추출
        #   이번 layer의 넓이(kernel 종류의 개수)
        width_now_layer = conv_width[i]
        #   이번 layer에 존재하는 모든 kernel의 크기 정보가 들어있는 행을 추출
        kernel_size_now_layer = kernel_size[i]
        #   직전 layer에 존재하는 모든 kernel의 개수 정보가 들어있는 행을 추출
        kernel_nums_bef_layer = kernel_num[i-1]
        #   이번 layer에 존재하는 모든 kernel의 개수 정보가 들어있는 행을 추출
        kernel_nums_now_layer = kernel_num[i]


        ##  맨 처음 conv layer의 input depth는 input data의 depth와 같아야 하고, 이 후부터는 직전 layer의 output depth와 같아야 함으로 if로 구분
        if i == 0 :
            input_depth = input_data_depth
        else :
            input_depth = sum(kernel_nums_bef_layer)

        ##  설정한 해당 layer의 input depth를 이번 layer에 존재하는 kernel의 개수 정보가 담긴 리스트 변수 맨 앞에 추가
        kernel_nums_now_layer = [input_depth] + kernel_nums_now_layer

        ##  해당 layer에서 원하는 kernel의 종류(depth)만큼 kernel를 생성
        for j in range(width_now_layer) :

            ##  원하는 kernel의 크기와 개수에 맞는 초기화된 weight와 bias를 생성
            #   각 kernel마다 이름을 부여
            now_kernel_name = 'conv_' + str(i+1) + '_' + str(j+1)
            #   해당 kernel의 이름을 모든 weight(or bias) 이름을 기록해둘 list에 추가
            weight_names_list.append(now_kernel_name)
            #   해당 kernel의 초기화된 weight를 모든 weight의 값을 저장할 딕셔너리에 추가
            #weights_dic[now_kernel_name] = init_weight([1, kernel_size_now_layer[j], kernel_nums_now_layer[0], kernel_nums_now_layer[j+1]], weight_init_type)
            weights_dic[now_kernel_name] = init_weight([kernel_size_now_layer[j], kernel_nums_now_layer[0], kernel_nums_now_layer[j + 1]], weight_init_type)
            #   해당 kernel의 초기화된 bias를 모든 bias의 값을 저장할 딕셔너리에 추가
            biases_dic[now_kernel_name] = init_bias([kernel_nums_now_layer[j+1]], bias_init_type)

        ##  첫 fully-connected layer로 들어갈 때 일렬로 펴주는 과정에서 직전 layer의 output depth은 마지막 layer의 존재하는 모든 kernel들의 개수 합과 일치(merge 함수의 특징)
        if i == (conv_deep-1) :
            final_depth = sum(kernel_nums_now_layer[1:])


    ### Fully-connected layer에서 원하는 모양의 weight와 bias를 만드는 부분
    ##  맨 처음 fully-connected layer의 input은 마지막 convolutional layer(pooling을 쓰면 pooling layer)에서 나온 feature map들을 모두 1xN 형태로 쭉 펴주어야 하기 때문에 N을 계산
    for k in range(pool_num) :

        #   맨 처음 pooling layer의 input은 input data의 column 이므로
        if k == 0 :
            after_pool_size = input_data_col

        #   각 pooling layer의 output size를 계산
        if (after_pool_size % 2) == 0 :
            after_pool_size = math.ceil(1 + ((after_pool_size - pool_size) / pool_str))
        else :
            after_pool_size = math.ceil(0 + ((after_pool_size - pool_size) / pool_str))

    ##  계산된 맨 처음 fully-connected layer의 input의 크기를 fully-connected layer의 unit 개수 정보가 담긴 리스트 변수 맨 앞에 추가하고 class의 개수를 맨 뒤에 추가
    #   일렬로 쭉 편 상태의 길이
    fc_input_length = final_depth * input_data_row * after_pool_size
    #   Fully-connected layer의 unit 개수 정보가 담긴 리스트 변수 맨 앞에 추가
    fc_layers_units = [fc_input_length] + fc_layers_units + [class_num]

    ##  한 fully-connect layer부터 차근차근 weight와 bias 생성
    print('Fully-connect layer에 해당하는 weight와 bias를 생성하고 있습니다.')
    for m in trange(fc_layers_num) :

        ##  원하는 크기와 개수에 맞는 초기화된 weight와 bias를 생성
        #   각 layer의 weight(or bias)마다 이름을 부여
        now_layer_name = 'fc_' + str(m+1)
        #   해당 layer의 weight의 이름을 모든 weight(or bias) 이름을 기록해둘 list에 추가
        weight_names_list.append(now_layer_name)
        #   해당 layer의 초기화된 weight를 모든 weight의 값을 저장할 딕셔너리에 추가
        weights_dic[now_layer_name] = init_weight([fc_layers_units[m], fc_layers_units[m + 1]], weight_init_type)
        #   해당 layer의 초기화된 bias를 모든 bias의 값을 저장할 딕셔너리에 추가
        biases_dic[now_layer_name] = init_bias([fc_layers_units[m+1]], bias_init_type)


    ### 마지막 fully-connected layer를 지나 output layer로 갈 때의 weight와 bias 생성
    print('Output layer에 해당하는 weight와 bias를 생성하고 있습니다.')
    #   Output layer의 weight(or bias) 이름을 모든 weight(or bias) 이름을 부여
    now_layer_name = 'output_layer'
    #   Output layer의 weight 이름을 모든 weight(or bias) 이름을 기록해둘 list에 추가
    weight_names_list.append(now_layer_name)
    #   Output layer의 초기화된 weight를 모든 weight의 값을 저장할 딕셔너리에 추가
    weights_dic[now_layer_name] = init_weight([fc_layers_units[fc_layers_num], fc_layers_units[-1]], weight_init_type)
    #   Output layer의 초기화된 bias를 모든 bias의 값을 저장할 딕셔너리에 추가
    biases_dic[now_layer_name] = init_bias([fc_layers_units[-1]], bias_init_type)


    ### 설정한 weight와 bias의 값들에 맞춰 생성된 weight의 이름(list), weight의 값(dictionary), bias의 값(dictionary), 첫 fully-connected layer의 input의 길이를 리턴
    return weight_names_list, weights_dic, biases_dic, fc_input_length



### Feature extraction 부분인 convolutional layer와 pooling layer의 구조(fully-connected layer 이전까지)를 형성하기 위한 함수
def make_feature_extraction_part (CNN_input_data, dic_weight, dic_bias, list_weight_name,
                                  conv_layer_deep, conv_layer_width, pooling_loc, pooling_size, pooling_stride,
                                  featuremaps,
                                  dropout_ratio
                                  ) :

    '''

    Input
        CNN_input_data          :   CNN의 맨 처음 input으로 들어가는 data set
        dic_weight              :   원하는 조건에 맞게 초기화된 weight 딕셔너리
        dic_bias                :   원하는 조건에 맞게 초기화된 bias 딕셔너리
        list_weight_name        :   생성된 weight(or bias)들의 이름이 담긴 리스트

        conv_layer_deep         :   얼마나 많은 convolutional layer를 만들지 지정 ex) 3 : conv layer를 3개 만듬
        conv_layer_width        :   Convolutional layer들의 각 layer마다 kernel의 종류가 얼마나 되는지에 대한 정보가 담긴 리스트 (각 원소가 차례대로 각 conv layer의 kernel 종류의 수를 의미)
        pooling_loc             :   Pooling layer를 어느 convolutional layer(or inception) 이후에 넣을지 정보가 담긴 리스트
        pooling_size            :   Pooling layer의 filter size (모든 pooling layer의 filter size를 동일하게 설정)
        pooling_stride          :   Pooling layer의 stride 값 (모든 pooling layer의 stride를 동일하게 설정)

        featuremaps             :   각 layer(conv, pooling, fc 모두)를 통과한 feature maps을 저장할 딕셔너리

        dropout_ratio           :   Feature extraction에서 실시할 dropout의 비율

    Output
        featuremaps             :   Feature extraction 부분의 각 layer을 통과하여 얻어진 feature maps을 모두 저장한 딕셔너리
        now_layer_input         :   Feature extraction의 젤 마지막 부분의 output (classification 부분의 input에 넣어주기 위해)
        featuremap_names_list   :   각 featuremap을 저장한 딕셔너리 키의 이름(feature maps의 이름)을 별도로 저장한 list (지금까지는 feature extraction 부분의 이름들만 들어가 있음)

    '''


    ##  변수 초기화
    #   Layer가 깊어질 때 원하는 weight와 bias를 선택하기 위해, 이름이 저장된 리스트에서 이전 layer에서 마지막으로 사용한 이름이 몇 번째에 존재하는지를 저장할 변수 초기화
    last_choice_name = 0
    #   처음 layer의 input은 입력된 input data set
    now_layer_input = CNN_input_data
    #   원하는 convolutional layer(or inception) 이후에 pooling layer를 넣기 위해 필요한 카운티 변수 초기화
    pool_count = 0
    #   각 featuremap을 저장한 딕셔너리 키의 이름(feature maps의 이름)을 별도로 저장하기 위해 list 초기화
    featuremap_names_list = []


    ##  원하는 layer의 깊이와 넓이에 맞게 convolutional layer와 pooling layer 형성
    for i in trange(conv_layer_deep) :

        ##  변수 초기화
        #   각 layer에서 병렬로 연결되어 있는 feature map들에 해당하는 tensor들을 리스트 형태로 저장할 변수 (tf.concat 함수에 이것을 입력으로 넣어주면 자동으로 depth 방향으로 합쳐줌)
        #   매 layer마다 초기화를 시켜주어야 해당 layer에 있는 feature map들만 depth 방향으로 합쳐짐.
        featuremaps_list_per_inception = []

        ##  생성된 convolutional layer의 kernel를 이용하여 각 layer마다 원하는 넓이(width)를 가지는 feature extraction 구조를 생성
        #   현재 layer에 만들 kernel 종류
        now_layer_width = conv_layer_width[i]
        #   해당 layer의 width를 형성
        for j in range(now_layer_width) :

            ##  해당 kernel의 convolutional layer 부분
            #   Convolution
            after_conv = tf.nn.conv1d(now_layer_input, dic_weight[list_weight_name[(last_choice_name + j)]],
                                      stride=1, padding='SAME')
            # after_conv = tf.nn.conv2d(now_layer_input, dic_weight[list_weight_name[(last_choice_name + j)]],
            #                           strides=[1, 1, 1, 1], padding='SAME')
            #   해당 kernel의 feature map 이름을 지정
            now_featuremap_name = list_weight_name[(last_choice_name + j)] + '_conv'
            #   해당 kernel을 통해 만들어진 feature map을 설장한 이름으로 전체 feature maps을 저장할 딕셔너리에 저장
            featuremaps[now_featuremap_name] = after_conv
            #   해당 feature map의 이름을 별도로 저장
            featuremap_names_list.append(now_featuremap_name)

            ##  Activation function 적용
            #   ReLU 이용
            after_relu = tf.nn.relu((after_conv + dic_bias[list_weight_name[(last_choice_name + j)]]))
            #   해당 activation function의 feature map 이름을 지정
            now_featuremap_name = list_weight_name[(last_choice_name + j)] + '_relu'
            #   Activation function을 통해 만들어진 feature map을 설장한 이름으로 전체 feature maps을 저장할 딕셔너리에 저장
            featuremaps[now_featuremap_name] = after_relu
            #   해당 feature map의 이름을 별도로 저장
            featuremap_names_list.append(now_featuremap_name)

            ##  나중에 depth 방향으로 합치기 위해 하나의 리스트에 저장
            featuremaps_list_per_inception.append(after_relu)

        ##  한 layer 안에 있는 다른 kernel로 부터 나온 feature maps을 depth 방향으로 합침.
        #   tf.concat을 이용하여 합침.
        inception_output = tf.concat(featuremaps_list_per_inception, 2)
        #   해당 inception의 이름을 지정
        now_featuremap_name = 'Inception_' + str(i+1)
        #   합쳐진 feature map을 설정한 이름으로 전체 feature maps을 저장할 딕셔너리에 저장
        featuremaps[now_featuremap_name] = inception_output
        #   해당 feature map의 이름을 별도로 저장
        featuremap_names_list.append(now_featuremap_name)
        #   Pooling의 사용유무와 관계없이 dropout input으로 넣기 위해
        dropout_input = inception_output

        ##  Pooling layer
        if (pooling_loc[pool_count] == (i+1)) and (len(pooling_loc) > pool_count) :

            #   Max pooling 방식
            #after_pool = tf.nn.max_pool(inception_output, ksize=[1, 1, pooling_size, 1], strides=[1, 1, pooling_stride, 1], padding='VALID')
            after_pool = tf.layers.max_pooling1d(inception_output, pool_size=pooling_size, strides=pooling_stride, padding='VALID')
            #   해당 pooling layer의 이름ㅇ르 지정
            now_featuremap_name = 'Pooling_' + str(i+1)
            #   Pooling을 통해 줄어든 feature map을 설정한 이름으로 전체 feature maps을 저장할 딕셔너리에 저장
            featuremaps[now_featuremap_name] = after_pool
            #   해당 feature map의 이름을 별도로 저장
            featuremap_names_list.append(now_featuremap_name)
            #   그 다음 pooling layer가 어디에 위치할지 보기 위해 index에 해당하는 카운티 변수 1증가
            pool_count = pool_count + 1
            #   Pooling의 사용유무와 관계없이 dropout input으로 넣기 위해
            dropout_input = after_pool

        ##  설정한 비율로 dropout 실시
        after_dropout = tf.nn.dropout(dropout_input, dropout_ratio)

        ##  다음 layer로 넘어갈 때는 현재의 합쳐진 feature maps(pooling가 포함되면 pooling의 output)이 다음 layer의 input이므로 업데이트
        now_layer_input = after_dropout

        ##  이름이 저장된 리스트에서 이전 layer에서 마지막으로 사용한 이름이 몇 번째에 존재하는지를 저장하여 그 다음 layer에서는 그 다음 이름부터 불러올 수 있도록
        last_choice_name = last_choice_name + now_layer_width


    ##  주어진 조건에 맞게 형성된 feature extraction 부분의 feature maps과 classification 부분(fully-connected layer)의 입력으로 넣어줄 마지막 layer의 output, 각 feature maps의 이름들을 리턴
    return  featuremaps, now_layer_input, featuremap_names_list



### Classification 부분인 fully-conneted layer와 output layer 구조를 형성하기 위한 함수
def make_classification_part (last_layer_output, dic_weight, dic_bias, list_weight_name, featuremap_names_list,
                              featuremaps,
                              fc_layer_deep, fc_input_length, sum_conv_width, dropout_ratio,
                              class_reg_type, rnn_input_len
                              ) :

    '''

    Input
        last_layer_output       :   Feature extraction의 마지막 layer에서 나온 output (이것이 classification의 첫 fully-connected layer의 input으로 들어오기 때문에)
        dic_weight              :   원하는 조건에 맞게 초기화된 weight 딕셔너리
        dic_bias                :   원하는 조건에 맞게 초기화된 bias 딕셔너리
        list_weight_name        :   생성된 weight(or bias)들의 이름이 담긴 리스트
        featuremap_names_list   :   각 featuremap을 저장한 딕셔너리 키의 이름(feature maps의 이름)을 별도로 저장한 list (지금까지는 feature extraction 부분의 이름들만 들어가 있음)

        featuremaps             :   각 layer(conv, pooling, fc 모두)를 통과한 feature maps을 저장할 딕셔너리

        fc_layer_deep           :   Fully-connected layers의 개수
        fc_input_length         :   첫 fully-connected layer의 input을 넣어주려면 1xN 형태로 쭉 펴주어야 하는데, 이 N은 앞에 feature extraction이 어떠한 구조를 가지고 있느냐에 따라 다름.
        sum_conv_width          :   Classification을 위한 fully-connected layer와 output layer에 맞는 이름을 불어오기 위해 feature extraction에 사용된 convoultion width들의 총 합
        dropout_ratio           :   Classification에서 실시할 dropout의 비율

    Output
        featuremaps             :   Classification 부분의 각 fully-connected layer을 통과하여 얻어진 feature maps을 모두 저장한 딕셔너리
        final_output            :   Output layer의 결과
        featuremap_names_list   :   각 featuremap을 저장한 딕셔너리 키의 이름(feature maps의 이름)을 별도로 저장한 list (classification 부분도 들어가 있음.)

    '''


    ##  Fully-connected layer의 input으로 넣어주기 위해서는 직전 layer에서 나온 output을 1xN 형태로 길게 펴주어야 함.
    reshape_last_layer_output = tf.reshape(last_layer_output, [-1, fc_input_length])


    ##  변수 초기화
    #   사전에 만들어진 weight와 bias의 이름 중에서 classification을 위한 fully-connected layer와 output layer에 맞는 이름을 불어오기 위해 불러온 이름의 시작지점 설정
    last_choice_name = sum_conv_width
    #   처음 layer의 input은 입력된 input data set
    now_layer_input = reshape_last_layer_output
    print('맨 처음 fc layer에 들어갈 weight와 bias의 이름은 : ', list_weight_name[last_choice_name])

    ##  원하는 layer의 수와 각 layer당 units 수에 맞게 fully-connected layers를 생성
    for i in trange(fc_layer_deep) :

        ##  Fully-connected 실시
        fully_connected = tf.matmul(now_layer_input, dic_weight[list_weight_name[last_choice_name + i]]) + dic_bias[list_weight_name[last_choice_name + i]]
        #   해당 kernel의 feature map 이름을 지정
        now_featuremap_name = list_weight_name[(last_choice_name + i)] + '_fc'
        #   해당 kernel을 통해 만들어진 feature map을 설장한 이름으로 전체 feature maps을 저장할 딕셔너리에 저장
        featuremaps[now_featuremap_name] = fully_connected
        #   해당 feature map의 이름을 별도로 저장
        featuremap_names_list.append(now_featuremap_name)

        ##  ReLU(activation function) 실시
        after_relu = tf.nn.relu(fully_connected)
        #   해당 kernel의 feature map 이름을 지정
        now_featuremap_name = list_weight_name[(last_choice_name + i)] + '_ReLU'
        #   해당 kernel을 통해 만들어진 feature map을 설장한 이름으로 전체 feature maps을 저장할 딕셔너리에 저장
        featuremaps[now_featuremap_name] = after_relu
        #   해당 feature map의 이름을 별도로 저장
        featuremap_names_list.append(now_featuremap_name)

        #   Dropout
        after_dropout = tf.nn.dropout(after_relu, dropout_ratio)
        #   그 다음 반복에서 사용할 input은 이번 fully-connected layer에서의 output에 해당
        now_layer_input = after_dropout


    if class_reg_type == 'CNN_classification' :
        ##  Output layer를 생성
        final_output = tf.matmul(now_layer_input, dic_weight['output_layer']) + dic_bias['output_layer']
        #   해당 kernel을 통해 만들어진 feature map을 설장한 이름으로 전체 feature maps을 저장할 딕셔너리에 저장
        featuremaps['output_layer'] = final_output
        #   해당 feature map의 이름을 별도로 저장
        featuremap_names_list.append('output_layer')

    elif class_reg_type ==  'fc_RNN_regression' :
        #   RNN의 input으로 넣어주어야 하기 때문에 크기를 수정
        final_output = tf.reshape(after_dropout, [-1, rnn_input_len, 1])


    ##  주어진 조건에 맞게 형성된 classification 부분의 feature maps과 output layer의 값, 각 feature maps의 이름들을 리턴
    return  featuremaps, final_output, featuremap_names_list
















### Convolutional Neural Network의 전체 구조를 완성하는 함수
def make_cnn_architecture (data_set,
                           weights_dic, biases_dic, weight_names_list,
                           conv_deep, conv_width, pool_location, pool_size, pool_str,
                           fc_layers_num, fc_input_len,
                           conv_dropout,fc_dropout,
                           choice_part, rnn_input_length
                           ) :

    '''

    Input
        data_set            :   CNN에 들어가는 input data set

        weights_dic         :   모든 weight의 값이 weight 이름을 key로 가지며 저장된 딕셔너리
        biases_dic          :   모든 bias의 값이 weight 이름을 key로 가지며 저장된 딕셔너리
        weight_names_list   :   모든 weight(or bias)의 이름들이 저장된 리스트

        conv_deep           :   얼마나 많은 convolutional layer를 만들지 지정 ex) 3 : conv layer를 3개 만듬
        conv_width          :   Convolutional layer들의 각 layer마다 kernel의 종류가 얼마나 되는지에 대한 정보가 담긴 리스트 (각 원소가 차례대로 각 conv layer의 kernel 종류의 수를 의미)
        pool_location       :   Pooling layer를 어느 convolutional layer(or inception) 이후에 넣을지 정보가 담긴 리스트
        pool_size           :   Pooling layer의 filter size (모든 pooling layer의 filter size를 동일하게 설정)
        pool_str            :   Pooling layer의 stride 값 (모든 pooling layer의 stride를 동일하게 설정)

        fc_layers_num       :   Fully-connected layers의 개수
        fc_input_len        :   첫 fully-connected layer의 input을 넣어주려면 1xN 형태로 쭉 펴주어야 하는데, 이 때의 N값

        conv_dropout        :   Feature extraction에서 실시할 dropout의 비율
        fc_dropout          :   Classification에서 실시할 dropout의 비율

    Output
        all_featuremaps     :   Feature extraction과 classification 부분에서 생성된 모든 feature maps이 저장되어 있는 딕셔너리
        py_x                :   Output layer의 값
        featuremaps_names   :   각 featuremap을 저장한 딕셔너리 키의 이름(feature maps의 이름)을 별도로 저장한 list (classification 부분도 들어가 있음.)

    '''

    ##  Feature extraction 부분
    if choice_part[0] == 'CNN_feature' :
        ##  Convolutional layer를 제작
        #   Convolutional layer의 맨 처음 input은 input data
        input_data = data_set
        #   모든 convolutional layer(pooling layer 포함)의 feature maps을 확인하기 위해 output을 모두 저장할 딕녀서리 초기화
        #   추가적으로 fully-connected layer의 feature maps도 저장
        all_featuremaps = {}
        #   Classification을 위한 fully-connected layer와 output layer에 맞는 이름을 불어오기 위해 feature extraction에 사용된 convoultion width들의 총 합을 저장
        fc_name_start_point = sum(conv_width)


        ##  Feature extraction 부분에 해당하는 convolutional layer, pooling layer의 구조 형성 (fully-connected layer 이전까지)
        print('CNN 중 feature extraction 부분의 구조를 생성하고 있습니다.')
        all_featuremaps, input_data, featuremaps_names = make_feature_extraction_part(input_data, weights_dic, biases_dic, weight_names_list,
                                                                                      conv_deep, conv_width, pool_location, pool_size, pool_str,
                                                                                      all_featuremaps,
                                                                                      conv_dropout)


    ##  Classification이나 regression을 진행하는 부분
    if choice_part[1] == 'CNN_classification' :
        ##  Classification 부분에 해당하는 fully-connected layer, output layer의 구조 형성
        print('CNN 중 classification 부분의 구조를 생성하고 있습니다.')
        all_featuremaps, py_x, featuremaps_names = make_classification_part(input_data, weights_dic, biases_dic, weight_names_list, featuremaps_names,
                                                                            all_featuremaps,
                                                                            fc_layers_num, fc_input_len, fc_name_start_point, fc_dropout,
                                                                            choice_part[1], rnn_input_length)
    elif choice_part[1] == 'fc_RNN_regression' :

        ##  CNN과 RNN 사이의 fully-connected는 한 번만 시행하는데 설정이 한 번이 아닌 경우 경고문을 출력
        if not fc_layers_num == 1 :
            print('Fully connected layer의 값이 1인지 확인해주세요!')

        ##  CNN-Fullyconnected-RNN에서 fully-connected 부분을 생성
        print('CNN에서 RNN으로 넘어갈 때 fully connected를 해주고 있습니다.')
        all_featuremaps, py_x, featuremaps_names = make_classification_part(input_data, weights_dic, biases_dic, weight_names_list, featuremaps_names,
                                                                            all_featuremaps,
                                                                            fc_layers_num, fc_input_len, fc_name_start_point, fc_dropout,
                                                                            choice_part[1], rnn_input_length)


    ##  모든 feature maps이 저장된 딕셔너리와 최종 output layer의 값을 리턴
    return  all_featuremaps, py_x, featuremaps_names










### 편집된 data와 형성된 CNN 구조를 가지고 학습을 진행하기 위한 함수
def cnn_training (train_data, train_num_label, weights_dic, biases_dic, all_featuremaps,
                  input_data_col, input_data_depth,
                  py_x, X, Y, p_keep_conv, p_keep_hidden, learning_rate, epoch_num, batch_size,
                  wanted_conv_dropout, wanted_fc_dropout,
                  test_data, test_num_label
                  ) :

    '''

    Input
        train_input_data            :   Train에 사용할 input data
        input_data_col              :   Input data에서 한 data의 행렬에서 column의 크기
        input_data_depth            :   Input data의 depth로 맨 처음 conv layer가 input으로 받는 depth를 설정해주기 위해 필요

    Output



    '''

    ##


    ##  CNN의 입력으로 사용할 변수들 설정 및 초기화
    #   Train용 data를 설정하고 실제 사용하기 위해 형성된 CNN 구조에 맞는 input data 형태로 변환
    trX = train_data
    trX = trX.reshape(-1, input_data_col, input_data_depth)
    #   Train용 label을 설정
    trY = train_num_label

    #####   Epoch의 overfitting 방지 지점 찾기 위해 넣은 부분
    ##  CNN의 입력으로 사용할 변수들 설정 및 초기화
    #   Test용 data를 설정하고 실제 사용하기 위해 형성된 CNN 구조에 맞는 input data 형태로 변환
    teX = test_data
    teX = teX.reshape(-1, input_data_col, input_data_depth)
    #   Test용 label을 설정
    teY = test_num_label
    #####


    ##  CNN 오차 최소화 관련된 함수
    #   오차(cost) 계산 (나중에 확률을 띄우고자 할 때에는 이부분에서 softmax와 logist를 분리해서 해야함!!!!!)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    p = tf.nn.softmax(py_x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y))
    # Loss function using L2 Regularization
    # for g in range(0, len(weight_names_list)) :
    #    if g == 0 :
    #        regularizer = tf.nn.l2_loss(weights_dic[weight_names_list[g]])
    #    else :
    #        regularizer = regularizer + tf.nn.l2_loss(weights_dic[weight_names_list[g]])
    # cost = tf.reduce_mean(cost + 0.01 * regularizer)
    # train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    ##  정확도 계산
    correct_pred = tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    ##  모든 Variables 변수 초기화
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)



########
    ##  Train에 대한 정보를 확인하기 위해 필요한 변수들 초기화
    #   매 에폭당 평균 정확도를 저장
    accuracies_train = []
    #   학습된 weights와 biases를 저장할 dictionary 변수 초기화
    weight_vals = {} ; biases_vals = {}
    #   모든 layers의 feature maps을 저장할 dictionary 변수 초기화
    train_layers_result = {}
    #   마지막 epoch의 결과를 저장할 list 변수 초기화
    #final_results = {}
    #   마지막 epoch의 정확도를 저장할 dictionary 변수 초기화
    #final_accuracy = {}
    #####   Epoch의 overfitting 방지 지점 찾기 위해 넣은 부분
    accuracies_test = []


    ##  Training 되기 전 weights와 biases 값을 저장
    ini_weights_val, ini_biases_val = sess.run([weights_dic, biases_dic], feed_dict={X: trX, Y: trY, p_keep_conv: 1.0, p_keep_hidden: 1.0})

    ##  Training하는 부분
    for i in trange(epoch_num) :

#############
        #   한 에폭당 batch가 몇개 인지 카운팅 하는 변수
        batch_times = 0
        #   한 에폭당 매 batch당 나온 accuracy를 저장
        accuracy_sum_per_epoch = 0
        #####   Epoch의 overfitting 방지 지점 찾기 위해 넣은 부분
        test_accuracy_sum_per_epoch = 0

        ##  입력한 batch_size에 맞게 input data를 분할
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX) + 1, batch_size))

        for start, end in training_batch:
            #   Start, end의 값이 어떤 식으로 들어가나 확인하기 위한 부분
            # print(start, end)

            ##  한 epoch당 몇 번의 batch가 실행되었는지 카운트 (나중에 accuracy 평균을 내는데 사용됨.)
            batch_times = batch_times + 1

            ##  Training하는 부분
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: (1 - wanted_conv_dropout), p_keep_hidden: (1 - wanted_fc_dropout)})

            ##  매 batch당 loss와 accuracy를 확인하기 위해 저장
            loss_per_batch, accuracy_per_batch = sess.run([cost, accuracy], feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: 1.0, p_keep_hidden: 1.0})
            #   Batch 당 나온 accuracy를 계속 더해줌. (나중에 batch 횟수로 나눠주면 한 epoch당 accuracy 평균이 나옴.)
            accuracy_sum_per_epoch = accuracy_sum_per_epoch + accuracy_per_batch

            ##  마지막 epoch에서만 실제 label과 예측된 label을 비교
            if i == (epoch_num-1) :
                if batch_times == 1 :
                    print()
                # 분류 정확히 하였나 직접 비교
                print(str(train_data.shape[0] // batch_size), ' 중에서 ', batch_times, ' 번째 실제 class 는   : ', (sess.run(tf.argmax(Y,1), feed_dict={Y: trY[start:end]})) + 1)
                print(str(train_data.shape[0] // batch_size), ' 중에서 ', batch_times, ' 번째 계산된 class 는 : ', (sess.run(tf.argmax(py_x,1), feed_dict={X: trX[start:end],  p_keep_conv: 1.0, p_keep_hidden: 1.0}) + 1))


        ##  계속 더해진 accuracy를 batch 횟수로 나눠 accuracy의 평균을 계산하여 저장
        #   한 epcoh 당 평균 accuracy를 계산
        acc_avg_train = accuracy_sum_per_epoch / batch_times
        #   계산된 accuracy를 저장할 리스트에 저장
        accuracies_train.append(acc_avg_train)
        #   평균 정확도를 확인
        print() ; print(str(i+1) + '번째 epoch에서의 정확도는 %.4f 입니다.' %acc_avg_train)

        #####   Epoch의 overfitting 방지 지점 찾기 위해 넣은 부분
        loss_per_batch, test_accuracy_per_batch = sess.run([cost, accuracy], feed_dict={X: teX, Y: teY, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        accuracies_test.append(test_accuracy_per_batch)

    ##  학습이 다 된 후 확인이 필요한 변수들 저장 (weights, biases, 각 layer의 feature maps, 최종 예측 output)
    weight_vals, biases_vals, train_layers_result, output_pred = sess.run([weights_dic, biases_dic, all_featuremaps, tf.argmax(py_x, 1)], feed_dict={X: trX, Y: trY, p_keep_conv: 1.0, p_keep_hidden: 1.0})

    ##  각 epoch당 구해진 평균 정확도를 그려서 변화 확인
    plt.figure(1)
    plt.plot(accuracies_train, 'blue')
    plt.plot(accuracies_test, 'red')
    plt.title('Accuracy per epoch (Train accuracy and test accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    ##
    return  accuracies_train, train_layers_result, weight_vals, biases_vals, output_pred.reshape(len(output_pred), 1), ini_weights_val, ini_biases_val





### CNN에서 cross-validation과 test 하는 함수
def cnn_test (test_data, test_num_label, all_featuremaps,
              input_data_col, input_data_depth, batch_size,
              py_x, X, Y, p_keep_conv, p_keep_hidden) :



    ##  CNN의 입력으로 사용할 변수들 설정 및 초기화
    #   Test용 data를 설정하고 실제 사용하기 위해 형성된 CNN 구조에 맞는 input data 형태로 변환
    teX = test_data
    teX = teX.reshape(-1, input_data_col, input_data_depth)
    #   Test용 label을 설정
    teY = test_num_label


    ##  CNN 오차 최소화 관련된 함수
    #   오차(cost) 계산 (나중에 확률을 띄우고자 할 때에는 이부분에서 softmax와 logist를 분리해서 해야함!!!!!)
    p = tf.nn.softmax(py_x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y))

    ##  정확도 계산
    correct_pred = tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    ## test 작동!!
    #   설정한 batch size를 바탕으로 test data의 구간을 나눔.
    testing_batch = zip(range(0, len(teX), batch_size),
                        range(batch_size, len(teX) + 1, batch_size))


    sess = tf.Session()

    ##  Test에 대한 정보를 확인하기 위해 필요한 변수들 초기화
    #   매 에폭당 평균 정확도를 저장
    accuracies_test = []


    acc_xaxis_test = 0
    pred = []

    for i in trange(1):

        #   한 에폭당 batch가 몇개 인지 카운팅 하는 변수
        batch_times = 0
        #   한 에폭당 매 batch당 나온 accuracy를 저장
        accuracy_sum_per_epoch = 0

        for start, end in testing_batch:
            #   Start, end의 값이 어떤 식으로 들어가나 확인하기 위한 부분
            # print(start, end)

            ##  한 epoch당 몇 번의 batch가 실행되었는지 카운트 (나중에 accuracy 평균을 내는데 사용됨.)
            batch_times = batch_times + 1

            ##  Test set에 대한 batch 당 정확도를 저장
            loss_per_batch, accuracy_per_batch = sess.run([cost, accuracy], feed_dict={X: teX[start:end], Y: teY[start:end], p_keep_conv: 1.0, p_keep_hidden: 1.0})
            accuracy_sum_per_epoch = accuracy_sum_per_epoch + accuracy_per_batch

            ##  실제 label과 예측된 label을 비교
            print(str(test_data.shape[0] // batch_size), ' 중에서 ', batch_times, ' 번째 실제 class 는   : ', sess.run(tf.argmax(Y, 1), feed_dict={Y: teY[start:end]}))
            print(str(test_data.shape[0] // batch_size), ' 중에서 ', batch_times, ' 번째 계산된 class 는 : ', sess.run(tf.argmax(py_x, 1), feed_dict={X: teX[start:end], p_keep_conv: 1.0, p_keep_hidden: 1.0}))




            #   Softmax만 쓴 경우로, class별 확률값을 보기위해
            #p_val = sess.run(p, feed_dict={X: teX[start:end], p_keep_conv: 1.0, p_keep_hidden: 1.0})
            '''
            for v in range(0,num_total_test) :
                pred.append(p_val[v])
                print(p_val[v])
            '''

            # 한번의 cross validation에서 생성된 confusion matrix 값
            #confusion_matrix_each_validation = make_confusion_matrix(desired_label, output_label)
            #confusion_matrix_each_validation = np.array(confusion_matrix_each_validation)

            '''
            print("batch_iter " + str(acc_xaxis_test) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Testing Accuracy= " + \
                  "{:.5f}".format(acc))
            '''

        ##  계속 더해진 accuracy를 batch 횟수로 나눠 accuracy의 평균을 계산하여 저장
        #   한 epcoh 당 평균 accuracy를 계산
        acc_avg_test = accuracy_sum_per_epoch / batch_times
        #   계산된 accuracy를 저장할 리스트에 저장
        accuracies_test.append(acc_avg_test)
        #   정확도를 확인

        print(); print('Test의 정확도는 %.4f 입니다.' % acc_avg_test)


    ##  Test의 각 layer를 통과한 feature maps과 최종 예측 output을 확인하기 위해 저장
    test_layers_result, output_pred = sess.run([all_featuremaps, tf.argmax(py_x, 1)], feed_dict={X: teX, Y: teY, p_keep_conv: 1.0, p_keep_hidden: 1.0})


    ##
    return  accuracies_test, test_layers_result, output_pred.reshape(len(output_pred), 1) #, confusion_matrix_each_validation

















