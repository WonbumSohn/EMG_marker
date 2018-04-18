#####---(Deep Learning을 이용한 프로그램)---#####
##  설명
#   Deep learning에 사용하여 원하는 알고리즘을 통해 원하는 타입의 학습을 진행하는 main 부분

##  업데이트 기록지
#   2018.03.19.월요일 : EMG 4채널 data와 28개의 marker 좌표(x,y,z,az,el,r) data를 불러와
#                      EMG 4채널 data로 부터 우선 1개의 marker 좌표(x,y,z,az,el,r)를 regression하는 알고리즘 구현 시작
#   2018.03.26.월요일 : Marker 값을 1000개의 구간으로 나눠 label을 만드는 것까지 완성
#                      다시 값(대표값)으로 바꿨을 때 거의 차이가 없었음.
#   2018.03.28.수요일 : GPU설치로 tensorflow를 gpu로 돌리는 것으로 체인지!
#   2018.04.04.수요일 : 필요한 files의 data만 불러오도록 수정 완료!


#####------------------------------------------------


### 사용할 lib 소환
from data_preparation import *
from CNN_algorithms import *
from data_representation import *

# 이미지를 출력하거나 plot을 그리기 위해
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

### Deep learning을 위한 data를 준비하면서 필요한 고정 변수들
##  Files의 이름과 주소를 불러오는 함수에 사용됨.
#   Data들이 들어있는 폴더의 주소
main_data_location = 'D:\\WonBumSohn\\data\\'
#   불러올 파일이 든 폴더 이름
experiment_name = 'Data_for_LSTM'
#   불러올 파일의 확장자명
want_file_type = '.csv'

##  사용하고자 하는 마커와 피실험자를 선택하는 함수에 사용됨.
#   전체 markers 중에서 몇 개의 markers를 !!연달아!! 선택할지와 무슨 makrer부터 선택할지([marker 선택 시작점(1 : 첫번재 마커, 2 : 두번째 마커, ...), marker 개수])
chosen_markers = [1, 1]
#   전체 피실험자 중 이번 deep learning에 사용할 피실험자 수와 어떤 피실험자부터 !!연달아!! 선택할지([피실험자 선택 시작점(1 : 첫번째 피실험자, 2 : 두번째 피실험자, ...), 피실험자 수]) (marker 개수가 !!여러개!!면 자동으로 [1, total_subjects_num]
chosen_subjects = [1, 1]

##  선택된 markers와 피실험자들에 맞는 data를 불러오는 함수에 사용됨.
#   불러올 파일들의 열의 이름
data_columns_name = ['EMG_ch1', 'EMG_ch2', 'EMG_ch3', 'EMG_ch4', 'marker_x', 'marker_y', 'marker_z', 'marker_az', 'marker_el', 'marker_r']

##  불러온 data를 input data와 target data로 나누는 함수에 사용됨.
#   EMG channel의 개수
EMG_chs_num = 4
#   실제 사용할 marker 좌표 ([시작좌표(순서는 1: x, 2: y, 3 : z), 사용할 좌표 개수) !!연달아!!
wanted_markers = [1, 1]

##  나눈 input data와 target data를 원하는 시퀀스 길이로 맞추는 함수에 사용됨. (CNN의 input 크기는 고정되어야 함.)
#   1개 시퀀스의 길이 (시계열데이터 입력 개수)
seq_length = 30

##  원하는 조건에 맞게 files의 순서를 섞는 함수에 사용됨.
#   Marker의 개수
total_markers_num = 28
#   피실험자 수
total_subjects_num = 21
#   실험 반복 횟수
subject_trial_num = 15
#   Train data를 뽑는 옵션 (0 : 모든 피실험자에서 동일한 수의 data를 train data로 추출, 1 : 피실험자를 고려하지 않고 정말 랜덤하게 train data로 추출)
#   Train data를 뽑는 방식이 다르기 때문에 섞을 때에도 섞는 방식이 다름 (0 : 해당 피실험자 내에서만 섞음, 1 : 전체 피실험자에서 섞음)
train_choice_option = 'same'
#train_choice_option = 0

##  Train data set과 test data set을 만드는 함수에 사용됨.
#   얼마만큼을 train set으로 쓸지 비율 입력
train_ratio = 0.7

##  Target data를 가지고 classification을 위해 label data를 만드는 함수에 사용됨.
#   Classification을 위해 나누고자 하는 class 수
x_axis_class_numer = 10


### 원하는 구조의 deep learning architecture를 만들고 학습하고 테스트하기 위해 필요한 고정 변수들
##
#   EMG 한 신호의 행 길이
EMG_row= 1
#   Marker의 좌표 개수
#marker_axis_num = 3

##  CNN을 돌리기 위한 변수들
#
weight_init_type = 1
#
bias_init_type = 1

#   Convolutional layer의 수 (얼마나 deep한지를 결정)
conv_layer_number = 1
#   한 개의 convolutional layer마다 width를 어디까지 늘려서 정확도를 확인할지 지정
conv_width_range = np.arange(1,(5 + 1), 1)

#   Kernel의 size를 어디까지 늘려서 정확도를 확인할지 지정 ([시작 , 끝])
#conv_kernel_size_range = [3, 7]
# #conv_kernel_size_range = [3, 9, 15, 19, 23] #np.arange(3, 25, 2)
conv_kernel_size_range = [1, 3, 5]
#   Kernel의 개수를 어디까지 늘려서 정확도를 확인할지 지정 ([시작, 끝])
conv_kernel_num_range = [16, 32, 64] #[8, 16, 32, 64, 128]

#
pooling_location = [1]
pooling_layer_number = len(pooling_location)
#
pooling_size = 1
#
pooling_stride = 1

#
fullyconnected_layer_number = 1
#0
fullyconnected_layer_unit = [100] #[1024, 1024]
#fullyconnected_layer_unit = [seq_length]

#
conv_dropout_ratio = 0.0
fc_dropout_ratio = 0.5
#
wanted_parts = ['CNN_feature', 'CNN_classification']
#wanted_parts = ['CNN_feature', 'fc_RNN_regression']


learning_speed = 1e-4
epoch_num = 300
batch_size = 20




### Data 불러오기
##  지정된 폴더 안에 있는 모든 하위 폴더에 든 data files의 주소를 한 번에 딕셔너리 형태로 저장 (key는 파일이름)
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100) ; print('해당 실험과 관련된 폴더 내에 존재하는 모든 files의 name을 저장하고 있습니다.') ; print()
total_files_location, total_files_name = find_file_address (main_data_location, experiment_name, want_file_type)
#files_list = find_files(experiment_name, want_file_type)
#   관련된 정보 출력
print()
print('지정된 폴더 안에 있는 모든 data files의 개수는(A) : ', len(total_files_location))
print('전체 data files의 name이자 key 이름 개수는(A와 같아야 함) : ', len(total_files_name))
print('전체 data files의 name은 : ') ; print(total_files_name)



### 이번 deep learning에서 사용할 피실험자들 data 선택
##  File(딕셔너리 key)의 이름을 저장해놓은 list에서 추출하면 딕셔너리에서 추출하는 효과 발생
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100) ; print('이번 deep learning에서 사용할 markers와 피실험자들의 files name을 추출하고 있습니다.') ; print()
chosen_files_name, chosen_subjects = choose_data (total_files_name, chosen_markers[0], chosen_markers[1], chosen_subjects[0], chosen_subjects[1], total_subjects_num, subject_trial_num, train_choice_option)
#   관련된 정보 추출력
print()
print('선택한 files name의 개수는(B) : ', len(chosen_files_name))
print('선택한 files name은 : ')
print(chosen_files_name)



### 선택된 files의 name을 가지고 원하는 files의 data만 불러오기
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100) ; print('위에서 추출한 files name을 가지고 해당하는 files의 data만 불러오고 있습니다.') ; print()
total_data = load_data(total_files_location, chosen_files_name, data_columns_name)
print()
print('불러온 data의 개수는(B와 같아야 함) : ', len(total_data))
print('첫 번째 data의 형태는 : ', type(total_data[chosen_files_name[0]]))
print('첫 번째 data의 크기는(C) : ', total_data[chosen_files_name[0]].shape)
print('첫 번째 data는 : ')
print(total_data[chosen_files_name[0]])
print('전체 data는 : ')
print(total_data)



### 불러온 data에서 input data와 target data(output의 기준)로 열을 나눔. (이때 몇 개의 축을 선택할지 고름)
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100) ; print('불러온 data에서 input data와 target data를 나누고 있습니다.') ; print()
input_data_set, target_data_set = separate_data (total_data, chosen_files_name, EMG_chs_num, wanted_markers)
print()
print('Input data의 총 개수는(B와 같아야 함) : ', len(input_data_set))
print('Target data의 총 개수는(B와 같아야 함) : ', len(target_data_set))
print(chosen_files_name[0], ' 파일 안에 든 input data의 측정 크기는(행의 개수는 C와 열의 개수는 EMG_chs_num과 같아야 함) : ', (input_data_set[chosen_files_name[0]]).shape)
print(chosen_files_name[0], ' 파일 안에 든 input data의 tpye은 : ', type(input_data_set[chosen_files_name[0]]))
print(chosen_files_name[0], ' 파일 안에 든 input data는 : ')
print(input_data_set[chosen_files_name[0]])
print(chosen_files_name[0], ' 파일 안에 든 target data의 측정 크기는(행의 개수는 B와 열의 개수는 사용하고자한 markers의 좌표 개수와 같아야 함) : ', (target_data_set[chosen_files_name[0]]).shape)
print(chosen_files_name[0], ' 파일 안에 든 target data의 tpye은 : ', type(target_data_set[chosen_files_name[0]]))
print(chosen_files_name[0], ' 파일 안에 든 target data는 : ')
print(target_data_set[chosen_files_name[0]])



### 원하는 시퀀스 길이에 맞게 data 편집
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100) ; print('원하는 시퀀스 길이에 맞게 data를 편집하고 있습니다. (CNN의 input은 크기를 맞춰주어야 하기 때문에) ') ; print()
adjusted_input_set, adjusted_target_set, adjusted_length = adjusted_sequence (input_data_set, target_data_set, chosen_files_name, seq_length)
print()
print('원하는 시퀀스 길이로 편집된 input data의 총 개수는(B와 같아야 함) : ', len(adjusted_input_set))
print('원하는 시퀀스 길이로 편집된 target data의 총 개수는(B와 같아야 함) : ', len(adjusted_target_set))
print('원하는 시퀀스 길이로 편집된 length의 총 개수(길이)는(B와 같아야 함) : ', len(adjusted_length))
print(chosen_files_name[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 input data의 길이는(D, 한 실험에 대한 CNN에 input으로 들어갈 data 개수) : ', len(adjusted_input_set[chosen_files_name[0]]))
print(chosen_files_name[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 input data의 tpye은 : ', type(adjusted_input_set[chosen_files_name[0]]))
print(chosen_files_name[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 input data의 shape은 : ', (adjusted_input_set[chosen_files_name[0]]).shape)
print(chosen_files_name[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 input data는 : ')
print(adjusted_input_set[chosen_files_name[0]])
print(chosen_files_name[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 target data의 길이는(D와 같아야 함) : ', len(adjusted_target_set[chosen_files_name[0]]))
print(chosen_files_name[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 target data의 tpye은 : ', type(adjusted_target_set[chosen_files_name[0]]))
print(chosen_files_name[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 target data의 shape은 : ', (adjusted_target_set[chosen_files_name[0]]).shape)
print(chosen_files_name[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 target data는 : ')
print(adjusted_target_set[chosen_files_name[0]])
print(chosen_files_name[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 input data의 길이는(D와 같아야 함) : ', (adjusted_length[chosen_files_name[0]]))
print('편집된 data들 중 1번째 마커, 1번재 피실험자의 data들 길이는 : ')
for i in range(subject_trial_num * chosen_subjects[1]) :
    if (i % subject_trial_num) == 0 :
        print(str((i//1) + 1), ' 번째 피실험자의 ', str(subject_trial_num), ' 번 반복 실험한 것의 편집 후 data 개수는 : ')
    print(adjusted_length[chosen_files_name[i]])



# ### Target을 classification의 label로 만드는 부분 (Classification을 위한 부분)
# #   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
# print('=' * 100) ; print('Classification을 위해 target data를 바탕으로 label data를 만드는 중입니다.') ; print()
# all_label, all_gap = make_num_label(adjusted_target_set, chosen_files_name, class_numer)
# print('Classification을 위해 label로 바꾼 딕셔너리의 총 key 개수는(A와 같아야 함) : ', len(all_label))
# print(chosen_files_name[0], ' 라는 key 안에 든 실제값의 간격은 : ', all_gap[chosen_files_name[0]])
# print(chosen_files_name[0], ' 라는 key 안에 든 label의 길이는(C, 한 실험에 대한 RNN에 input으로 들어갈 data 개수) : ', len(all_label[chosen_files_name[0]]))
# print(chosen_files_name[0], ' 라는 key 안에 든 label의 tpye은 : ', type(all_label[chosen_files_name[0]]))
# print(chosen_files_name[0], ' 라는 key 안에 든 label의 shape은 : ', (all_label[chosen_files_name[0]]).shape)
# print(chosen_files_name[0], ' 라는 key 안에 든 label은 : ')
# print(adjusted_target_set[chosen_files_name[0]][0])
# print(all_label[chosen_files_name[0]][0])



# ### Class를 다시 숫자로 바꾸는 부분
# #   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
# print('=' * 100)
# print('그래프를 그리기 위해 label을 다신 대표값으로 바꾸는 중')
# all_label_value = return_value (all_label, chosen_file_names, class_numer, all_gap)
# print('Label에 의거하여 대표값을 넣은 딕셔너리의 총 key 개수는(A와 같아야 함) : ', len(all_label_value))
# print(chosen_file_names[0], ' 라는 key 안에 든 대표값 배열의 길이는(C, 한 실험에 대한 RNN에 input으로 들어갈 data 개수) : ', len(all_label_value[chosen_file_names[0]]))
# print(chosen_file_names[0], ' 라는 key 안에 든 대표값 배열의 tpye은 : ', type(all_label_value[chosen_file_names[0]]))
# print(chosen_file_names[0], ' 라는 key 안에 든 대표값 배열의 shape은 : ', (all_label_value[chosen_file_names[0]]).shape)
# print(chosen_file_names[0], ' 라는 key 안에 든 대표값 배열은 : ')
# print(adjusted_target_set[chosen_file_names[0]][0:10])
# print(all_label_value[chosen_file_names[0]][0:10])
#
#
# ##  원래 값과 label을 다시 값으로 바꾼 대표값 사이의 차이 정도를 보기 위해 그래프 출력
# # plt.figure(1)
# # plt.plot(adjusted_target_set[chosen_file_names[0]], 'blue')
# # plt.plot(all_label_value[chosen_file_names[0]], 'red')
# # plt.show()



### 한 마커당 한 피실험자의 반복 실험을 통해 얻은 data의 순서를 랜덤하게 섞음.
##  File(딕셔너리 key)의 이름을 저장해놓은 list만 섞으면 딕셔너리를 자동으로 섞이는 효과 발생
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100) ; print('불러온 data의 순서를 원하는 방법으로 랜덤하게 섞고 있습니다.') ; print()
shuf_total_files_name = shuffle_data (chosen_files_name, chosen_markers[1], chosen_subjects[1], subject_trial_num, train_choice_option, total_subjects_num)
#   관련된 정보 출력
print()
print('섞인 후 files name의 개수는(B와 같아야 함) : ', len(shuf_total_files_name))
print('섞인 후 files name는 : ')
print(shuf_total_files_name)
print('섞인 후 files name을 체크')
print('첫번째 마커, 첫번째 피실험자 : ') ; print(shuf_total_files_name[(0 * subject_trial_num) : (1 * subject_trial_num)])
print('첫번째 마커, 두번째 피실험자 : ') ; print(shuf_total_files_name[(1 * subject_trial_num) : (2 * subject_trial_num)])



### 원하는 train 비율만큼 train set을 만들고 나머지는 test set을 만들기 위해 train 비율만큼 선택된 file name 중 일부를 train set으로 지정 후 나머지는 test set으로 지정
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100) ; print('원하는 train 비율만큼 train set용 data와 test set용 data을 만들고 있습니다.') ; print()
train_files_name, test_files_name = make_train_test_list_set (shuf_total_files_name, subject_trial_num, train_ratio, chosen_markers[1], chosen_subjects[1], train_choice_option)
print()
print('Train용 files name의 개수는 : ', len(train_files_name))
print('Train용 files name 는 : ')
print(train_files_name)
print('Test용 files name의 개수는 : ', len(test_files_name))
print('Test용 files name 는 : ')
print(test_files_name)

#   Classification 용
#train_data_set, train_label_set, test_data_set, test_label_set = make_train_test_set(adjusted_input_set, all_label, train_files_name, test_files_name)
#   Regression 용
print()
train_data_set, train_target_set, test_data_set, test_target_set = make_train_test_set(adjusted_input_set, adjusted_target_set, train_files_name, test_files_name)
print()
print('Train data의 type은 : ', type(train_data_set))
print('Train data의 크기는 : ', (train_data_set).shape)
print('Train data는 : ')
print(train_data_set)
print('Train target의 type은 : ', type(train_target_set))
print('Train target의 크기는 : ', (train_target_set).shape)
print('Train target은 : ')
print(train_target_set)
print('Test data의 type은 : ', type(test_data_set))
print('Test data의 크기는 : ', (test_data_set).shape)
print('Test data는 : ')
print(test_data_set)
print('Test target의 type은 : ', type(test_target_set))
print('Test target의 크기는 : ', (test_target_set).shape)
print('Test target은 : ')
print(test_target_set)

##  Data에서 어느 위치에 nan 값이 존재하는지 체크 (2번째 marker가 기준인 코 marker라 0이 아닌 nan 이 나옴.
#print('Train target set에 nan이 들어있는 위치는 : ') ; print(np.argwhere(np.isnan(train_target_set)))
#print('Test target set에 nan이 들어있는 위치는 : ') ; print(np.argwhere(np.isnan(test_target_set)))

##  Data의 최소, 최대값을 확인
#print('Train target data의 최소값은 : ', np.nanmin(train_target_set, axis=0)) ; print('Train target data의 최대값은 : ', np.nanmax(train_target_set, axis=0))
#print('Test target data의 최소값은 : ', np.nanmin(test_target_set, axis=0)) ; print('Test target data의 최대값은 : ', np.nanmax(test_target_set, axis=0))



### Target을 classification의 label로 만드는 부분 (Classification을 위한 부분)
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100) ; print('Classification을 위해 target data를 바탕으로 label data를 만드는 중입니다.') ; print()
train_label, test_label, train_target_minimum, train_target_maximum, train_gap_in_class = make_num_label(train_target_set, test_target_set, x_axis_class_numer)
print()
print('Train target의 최소값은 : ', train_target_minimum)
print('Train target의 최대값은 : ', train_target_maximum)
print('Class를 나눌 때 기준이 된 targap 값의 간격은 : ', train_gap_in_class)

print('Classification을 위해 label로 바꾼 train label의 tpye은 : ', type(train_label))
print('Classification을 위해 label로 바꾼 train label의 크기는 (행=train target의 행, 열=class 개수) : ', train_label.shape)
print('Train label 은 : ') ; print(train_label)
print('첫번째 train target 값은 : ') ; print(train_target_set[0])
print('첫번째 train label 은 : ') ; print(train_label[0])
print('첫번째 train label에서 1이 나온 위치는 : ', np.where(train_label[0]==1))

print('Classification을 위해 label로 바꾼 train label의 tpye은 : ', type(test_label))
print('Classification을 위해 label로 바꾼 test label의 크기는 (행=test target의 행, 열=class 개수) : ', test_label.shape)
print('Test label 은 : ') ; print(test_label)
print('첫번째 test target 값은 : ') ; print(test_target_set[0])
print('첫번째 test label 은 : ') ; print(test_label[0])
print('첫번째 test label에서 1이 나온 위치는 : ', np.where(test_label[0]==1))

##  원래 값과 label을 다시 값으로 바꾼 대표값 사이의 차이 정도를 보기 위해 그래프 출력
# for i in range(len(shuf_total_files_name)) :
#     plt.figure(i)
#     plt.ylim((train_target_maximum - (2 * train_gap_in_class))-0.02, train_target_maximum+0.02)
#     plt.axhline(y=(train_target_maximum - (0 * train_gap_in_class)), color='r', linewidth=1)
#     plt.axhline(y=(train_target_maximum - (1 * train_gap_in_class)), color='r', linewidth=1)
#     plt.axhline(y=(train_target_maximum - (2 * train_gap_in_class)), color='r', linewidth=1)
#     plt.plot(adjusted_target_set[shuf_total_files_name[i]], 'ok')
# plt.show()

# plt.figure(1)
# plt.plot(target_data_set[shuf_total_files_name[0]][(seq_length-1) : len(target_data_set[shuf_total_files_name[0]])], 'ok')
#
# plt.figure(2)
# plt.plot(target_data_set[shuf_total_files_name[0]][(seq_length-1) : len(target_data_set[shuf_total_files_name[0]])], 'black')
# plt.plot(adjusted_target_set[shuf_total_files_name[0]], 'blue')
# #plt.plot(train_target_set[0:adjusted_length[shuf_total_files_name[0]]], 'red')
#
# plt.figure(3)
# plt.plot(target_data_set[shuf_total_files_name[0]][(seq_length-1) : len(target_data_set[shuf_total_files_name[0]])], 'black')
# #plt.plot(adjusted_target_set[shuf_total_files_name[0]], 'blue')
# plt.plot(train_target_set[0:adjusted_length[shuf_total_files_name[0]]], 'red')
#
# plt.figure(4)
# plt.plot(adjusted_target_set[shuf_total_files_name[0]], 'blue')
# plt.plot(train_target_set[0:adjusted_length[shuf_total_files_name[0]]], 'red')
# plt.show()





#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100) ; print('=' * 100) ; print('이제부터는 본격적으로 CNN training에 들어갑니다.') ; print('=' * 100) ; print('=' * 100)


### 최적의 상태를 찾기 위해 for 구문을 시행
##  매 반복에서의 최종 train, test 정확도를 저장해기 위해 필요한 변수 초기화
#   Train 정확도를 저장하기 위한 list 변수 초기화
total_train_accuracies = []
#   Test 정확도를 저장하기 위한 list 변수 초기화
total_test_accuracies = []
#   Train과 test의 정확도 이름을 저장할 list 변수 초기화 (그래야 그래프에서 x축에 이름을 넣을 수 있음)
total_accuracy_names = []
#   Weight 초기화 방법에서 xavier 사용시 이름이 매번 달라야 해서 인위적으로 매 반복마다 업데이트 되는 정수 초기화
iter_order = 0

##  반복 시행
#for loop_width in conv_width_range :

    #itertools.product()

#for loop_size in range(conv_kernel_size_range[0], (conv_kernel_size_range[1] + 1)) :
for loop_size in conv_kernel_size_range :

    #for loop_num in range(conv_kernel_num_range[0], (conv_kernel_num_range[1] + 1)) :
    for loop_num in conv_kernel_num_range :

        #   Weight 초기화 방법에서 xavier 사용시 이름이 매번 달라야 해서 인위적으로 매 반복마다 업데이트
        iter_order = iter_order + 1

        print() ; print('Kernel size ', conv_kernel_size_range, '와 kernel 개수 ', conv_kernel_num_range, '중에서')
        print('[[[[[지금은 kernel size가 ', str(loop_size), '일 때 kernel의 개수가 ', str(loop_num), '일 때 입니다.]]]]]') ; print()


        #   한 개의 convolutional layer에서 몇 개의 서로 다른 filter를 사용하여 특징을 뽑아낼 것인지 지정
        #   각 원소는 차례대로 각 layer에 존재하는 서로 다른 filter의 개수를 의미
        conv_width_number = [1]

        #   Convolutional layer에 존재하는 모든 filter의 size 지정
        #   행 : 각 layer를 의미, 열 : 각 layer에 존재하는 서로 다른 filter들의 각각의 size를 의미
        # conv_kernel_size = [[3, 5, 7, 9, 11]]
        conv_kernel_size = [[loop_size]]

        #   Convolutional layer에 존재하는 모든 filter의 개수 지정
        #   행 : 각 layer를 의미, 열 : 각 layer에 존재하는 서로 다른 filter들의 각각의 개수를 의미
        # conv_kernel_number = [[32, 32, 32, 32, 32]]
        conv_kernel_number = [[loop_num]]


        ### Deep learning 구조를 만들고 학습하고 테스트하는데 필요한 변수지만 앞에서 지정한 값에 의해 자동으로 정해지는 변수들
        ##  나중에 training와 validation, test시 data와 label 입력을 위해 data type과 size 먼저 지정 (구체적인 값은 나중에 지정)
        #   Data를 입력할 변수의 data type과 size 지정 (None은 batch size에 따라 바뀌므로 특정한 값으로 지정하지 않은 것)
        #X = tf.placeholder("float", [None, 1, seq_length, EMG_ch_num])
        X = tf.placeholder("float", [None, seq_length, EMG_chs_num])
        #   Label을 입력할 변수의 label type과 size 지정 (None은 batch size에 따라 바뀌므로 특정한 값으로 지정하지 않은 것)
        Y = tf.placeholder("float", [None, x_axis_class_numer])

        ##  Dropout을 사용하기 위해 dropout의 변수 type을 설정 (역시 구체적인 값은 나중에 지정)
        #   Convolutional layer(ReLU, Pooling 다 포함한 용어)에 적용할 dropout type 지정
        p_keep_conv = tf.placeholder("float")
        #   Fully connected layer에 적용할 dropout type 지정
        p_keep_hidden = tf.placeholder("float")



        ### CNN을 돌리기 위해 필요한 weight와 bias들을 원하는 값에 맞춰 생성
        #   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
        print('=' * 100) ; print('설정한 weight와 bias의 크기와 사이즈, layer의 깊이와 넓이에 맞게 초기화된 weight와 bias들을 생성하기 시작하였습니다.') ; print()
        weight_names, weights, biases, fisrt_fc_input_len =  make_wei_bias (conv_layer_number, conv_width_number, conv_kernel_size, conv_kernel_number, pooling_layer_number, pooling_size, pooling_stride,
                                                                            fullyconnected_layer_number, fullyconnected_layer_unit,
                                                                            EMG_row, seq_length, EMG_chs_num, x_axis_class_numer,
                                                                            weight_init_type, bias_init_type , iter_order)
        print('초기화된 weight와 bias를 모두 생성하였습니다.') ; print()

        with tf.Session() as sess :
            init = tf.global_variables_initializer()
            sess.run(init)

            print('모든 weight(or bias)의 이름은 : ')
            print(weight_names)

            for i in range(len(weight_names)) :
                print(weight_names[i], '에 있는 weight의 type은 : ', type(sess.run(weights[weight_names[i]])))
                print(weight_names[i], '에 있는 weight의 크기는 : ', (sess.run(weights[weight_names[i]])).shape)
                #print(weight_names[i], '에 있는 weight의 값은 : ')
                #print((sess.run(weights[weight_names[i]])))

            for i in range(len(weight_names)):
                print(weight_names[i], '에 있는 bias의 type은 : ', type(sess.run(biases[weight_names[i]])))
                print(weight_names[i], '에 있는 bias의 크기는 : ', (sess.run(biases[weight_names[i]])).shape)
                #print(weight_names[i], '에 있는 bias의 값은 : ')
                #print((sess.run(biases[weight_names[i]])))

        print('첫 fully-connected layer의 input으로 들어가는 data의 길이는 : ', fisrt_fc_input_len)



        ### 초기화된 weight와 bias들을 이용하여 원하는 CNN 구조를 생성
        #   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
        print('=' * 100) ; print('초기화된 weight와 bias들을 이용하여 원하는 CNN 구조를 생성하기 시작하였습니다.') ; print()
        featuremaps, output_value, all_featuremaps_names = make_cnn_architecture (X,
                                                                                  weights, biases, weight_names,
                                                                                  conv_layer_number, conv_width_number, pooling_location, pooling_size, pooling_stride,
                                                                                  fullyconnected_layer_number, fisrt_fc_input_len,
                                                                                  p_keep_conv, p_keep_hidden,
                                                                                  wanted_parts, seq_length)

        #featuremaps_keys = list(featuremaps.keys())
        featuremaps_keys = all_featuremaps_names
        print('feature maps 딕셔너리에 있는 keys는 :') ; print(featuremaps_keys)
        for i in range(len(featuremaps_keys)) :
            print(featuremaps_keys[i], '안에 있는 feature map의 tensor는 : ', featuremaps[featuremaps_keys[i]])
        print('Output layer의 값은 : ') ; print(output_value)



        ### 형성된 CNN 구조를 가지고 training 시행
        print('=' * 100) ; print('형성된 CNN 구조를 이용하여 학습을 시작합니다.') ; print()
        train_accuracies, train_featuremaps, updated_weights, updated_biases, target_trainset_output, estimated_trainset_output, initial_weights, initial_biases = \
            cnn_training (train_data_set, train_label, weights, biases, featuremaps,
                          seq_length, EMG_chs_num,
                          output_value, X, Y, p_keep_conv, p_keep_hidden, learning_speed, epoch_num, batch_size,
                          conv_dropout_ratio, fc_dropout_ratio,
                          test_data_set, test_label)

        print('Train set에 대한 setimated label의 type은 ; ') ; print(type(estimated_trainset_output))
        print('Train set에 대한 estimated label의 크기는 : ') ; print(estimated_trainset_output.shape)

        ### Class를 다시 숫자로 바꾸는 부분 (학습 후 필요!!!!!!!!!!!!)
        #   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
        print('=' * 100) ; print('그래프를 그리기 위해 label을 다신 대표값으로 바꾸는 중') ; print()
        estimated_train_target_value = return_value (estimated_trainset_output, x_axis_class_numer, train_target_minimum, train_gap_in_class)
        print()
        print('Train set에서 estimated label에 의거하여 대표값을 대입한 결과의 type은 : ', type(estimated_train_target_value))
        print('Train set에서 estimated label에 의거하여 대표값을 대입한 결과의 크기는 (원래 train_target_set의 크기와 같으면 됨.) : ', (estimated_train_target_value).shape)
        print('첫번째 train target 값은 : ') ; print(train_target_set[0])
        print('첫번째 estimated target value는 : ') ; print(estimated_train_target_value[0])
        '''
        ##  원래 값과 label을 다시 값으로 바꾼 대표값 사이의 차이 정도를 보기 위해 그래프 출력
        for i in range(len(train_files_name)) :
            plt.figure(i)
            if i == 0 :
                plt_start = 0
                plt_end = adjusted_length[shuf_total_files_name[i]]
            else :
                plt_start = plt_start + adjusted_length[shuf_total_files_name[i-1]]
                plt_end = plt_end + adjusted_length[shuf_total_files_name[i]]
            plt.plot(train_target_set[plt_start: plt_end], 'blue')
            plt.plot(estimated_target_value[plt_start: plt_end], 'red')
            plt.title('Actual marker value vs Estimated marker value (1st marker, x-axis)')
            plt.xlabel('Adjusted Time (Real experiment time - window size)')
            plt.ylabel('Value (x-axis)')
        plt.show()
        '''

        ### 학습된 CNN 구조와 파라미터를 가지고 test 시행
        featuremaps, output_value, _ = make_cnn_architecture (X,
                                                              updated_weights, updated_biases, weight_names,
                                                              conv_layer_number, conv_width_number, pooling_location, pooling_size, pooling_stride,
                                                              fullyconnected_layer_number, fisrt_fc_input_len,
                                                              p_keep_conv, p_keep_hidden,
                                                              wanted_parts, seq_length)

        test_accuracies, test_featuremaps, target_testset_output, estimated_testset_output = \
            cnn_test (test_data_set, test_label, featuremaps,
                      seq_length, EMG_chs_num, batch_size,
                      output_value, X, Y, p_keep_conv, p_keep_hidden)

        print('Test set에 대한 setimated label의 type은 ; '); print(type(estimated_testset_output))
        print('Test set에 대한 estimated label의 크기는 : ') ; print(estimated_testset_output.shape)



        ### Class를 다시 숫자로 바꾸는 부분 (학습 후 필요!!!!!!!!!!!!)
        #   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
        print('=' * 100) ; print('그래프를 그리기 위해 label을 다신 대표값으로 바꾸는 중') ; print()
        estimated_test_target_value = return_value (estimated_testset_output, x_axis_class_numer, train_target_minimum, train_gap_in_class)
        print()
        print('Test set에서 estimated label에 의거하여 대표값을 대입한 결과의 type은 : ', type(estimated_test_target_value))
        print('Test set에서 estimated label에 의거하여 대표값을 대입한 결과의 크기는 (원래 train_target_set의 크기와 같으면 됨.) : ', (estimated_test_target_value).shape)
        print('첫번째 train target 값은 : ') ; print(train_target_set[0])
        print('첫번째 estimated target value는 : ') ; print(estimated_test_target_value[0])

        '''
        ##  원래 값과 label을 다시 값으로 바꾼 대표값 사이의 차이 정도를 보기 위해 그래프 출력
        for i in range(len(test_files_name)) :
            plt.figure(i)
            if i == 0 :
                plt_start = 0
                plt_end = adjusted_length[test_files_name[i]]
            else :
                plt_start = plt_start + adjusted_length[test_files_name[i-1]]
                plt_end = plt_end + adjusted_length[test_files_name[i]]
            plt.plot(test_target_set[plt_start: plt_end], 'blue')
            plt.plot(estimated_target_value[plt_start: plt_end], 'red')
        plt.show()
        '''

        #   매 반복당 최종 epoch에서의 정확도를 train과 test set 별로 저장
        total_train_accuracies.append(train_accuracies[-1])
        total_test_accuracies.append(test_accuracies[-1])
        now_accuracy_name = 'size_' + str(loop_size) + '_num_' + str(loop_num)
        total_accuracy_names.append(now_accuracy_name)





        import os
        ##  csv 파일로 저장
        #   저장할 폴더명 생성
        save_folder_name = './' + experiment_name + '_convdeep_' + str(conv_layer_number) + '_convwidth_' + str(conv_width_number[0]) +\
                           '_fcdeep_' + str(fullyconnected_layer_number) + '_fcunit_' + str(fullyconnected_layer_unit[0]) + '_epoch_' + str(epoch_num)

        #   설정한 이름의 폴더가 있는지 확인 후 없으면 생성
        if not os.path.isdir(save_folder_name) :
            os.mkdir(save_folder_name)

        ##  매 반복마다의 정확도를 저장
        #   파일 이름 지정
        save_file_name = save_folder_name + '/loop_accuracy_summary'
        save_file_name = save_file_name + '.csv'
        #   파일에 저장
        with open(save_file_name, 'at', newline = '') as csvfile :
            writer = csv.writer(csvfile, delimiter = ',')

            save_file_data = np.concatenate((train_accuracies[-1].reshape((1,1)), test_accuracies[-1].reshape((1,1))), axis=0)
            #save_file_data = np.transpose(save_file_data)

            # 맨 처음에만 열의 이름 부여
            if (loop_size == conv_kernel_size_range[0]) and (loop_num == conv_kernel_num_range[0]) :
                writer.writerow(['Loop_name', 'Train accuracy', 'Test accuracy'])

            writer.writerow([str('kernelsize_' + str(loop_size) + '_kernelnum_' + str(loop_num))] + save_file_data[0].tolist() + save_file_data[1].tolist())
        csvfile.close()


        ##  실제 마커 값과 예측한 마커값을 저장
        #   파일 이름 지정 (Train 경우)
        save_file_name = save_folder_name + '/kernelsize_' + str(loop_size) + '_kernelnum_' + str(loop_num) + '_trainaccuracy_' + str(train_accuracies[-1]) #+ '_testaccuracy_' + test_accuracies[-1]
        #save_file_name_feature = save_file_name + '_featuremaps'
        save_file_name = save_file_name + '_estimation'
        save_file_name = save_file_name + '.csv'

        #   파일에 저장
        # for i in range(conv_layer_number) :
        with open(save_file_name, 'at', newline = '') as csvfile :
            writer = csv.writer(csvfile, delimiter = ',')

            #   저장할 train data 편집
            for i in range(len(train_files_name)) :
                if i == 0 :
                    index_start = 0
                    index_end = adjusted_length[train_files_name[i]]
                else :
                    index_start = index_start + adjusted_length[train_files_name[i-1]]
                    index_end = index_end + adjusted_length[train_files_name[i]]

                #   저장된 길이를 가지고 계산한 배열의 시작점과 끝점을 가지고 원하는 부분의 data만 추출 후 전치
                actual_target_per_trial = train_target_set[index_start:index_end]
                actual_target_per_trial = np.transpose(actual_target_per_trial)
                estimated_target_per_trial = estimated_train_target_value[index_start:index_end]
                estimated_target_per_trial = np.transpose(estimated_target_per_trial)
                #   추출한 data를 행 병합
                save_file_data = np.concatenate((actual_target_per_trial, estimated_target_per_trial), axis=0)
                axis_num = int(save_file_data.shape[0] / 2)

                #   Target 부터 csv 파일에 기록
                for j in range(0, axis_num) :
                    writer.writerow([str(train_files_name[i] + ['_actual_x', '_actual_y', '_actual_z'][j])] + save_file_data[j].tolist())
                #   예측된 값을 csv 파일에 기록
                for j in range(axis_num, (axis_num*2)) :
                    writer.writerow([str(train_files_name[i] + ['_estimated_x', '_estimated_y', '_estimated_z'][j-axis_num])] + save_file_data[j].tolist())
        csvfile.close()

        #   파일 이름 지정 (Test 경우)
        save_file_name = save_folder_name + '/kernelsize_' + str(loop_size) + '_kernelnum_' + str(loop_num) + '_testaccuracy_' + str(test_accuracies[-1])  # + '_testaccuracy_' + test_accuracies[-1]
        # save_file_name_feature = save_file_name + '_featuremaps'
        save_file_name = save_file_name + '_estimation'
        save_file_name = save_file_name + '.csv'

        #   파일에 저장
        # for i in range(conv_layer_number) :
        with open(save_file_name, 'at', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            #   저장할 test data 편집
            for i in range(len(test_files_name)):
                if i == 0:
                    index_start = 0
                    index_end = adjusted_length[test_files_name[i]]
                else:
                    index_start = index_start + adjusted_length[test_files_name[i - 1]]
                    index_end = index_end + adjusted_length[test_files_name[i]]

                #   저장된 길이를 가지고 계산한 배열의 시작점과 끝점을 가지고 원하는 부분의 data만 추출 후 전치
                actual_target_per_trial = test_target_set[index_start:index_end]
                actual_target_per_trial = np.transpose(actual_target_per_trial)
                estimated_target_per_trial = estimated_test_target_value[index_start:index_end]
                estimated_target_per_trial = np.transpose(estimated_target_per_trial)
                #   추출한 data를 행 병합
                save_file_data = np.concatenate((actual_target_per_trial, estimated_target_per_trial), axis=0)
                axis_num = int(save_file_data.shape[0] / 2)

                #   Target 부터 csv 파일에 기록
                for j in range(0, axis_num):
                    writer.writerow([str(test_files_name[i] + ['_actual_x', '_actual_y', '_actual_z'][j])] + save_file_data[j].tolist())
                #   예측된 값을 csv 파일에 기록
                for j in range(axis_num, (axis_num * 2)):
                    writer.writerow([str(test_files_name[i] + ['_estimated_x', '_estimated_y', '_estimated_z'][j - axis_num])] + save_file_data[j].tolist())
        csvfile.close()



        ##  매 반복시마다 confusion matrix를 저장
        #   Confusion matrix의 label을 지정
        confusion_matrix_labels = np.arange(0+1, x_axis_class_numer+1)
        #   Train에 대한 confusion matrix 작성
        #save_file_name = save_folder_name + '/kernelsize_' + str(loop_size) + '_kernelnum_' + str(loop_num) + '_train_confusionmatrix'
        #confusion_matrix_plot(target_trainset_output, estimated_trainset_output.reshape(len(estimated_trainset_output)), confusion_matrix_labels, save_file_name, save_status=True, normalize=False, title='Confusion matrix of train data set')
        save_file_name = save_folder_name + '/kernelsize_' + str(loop_size) + '_kernelnum_' + str(loop_num) + '_train_confusionmatrix' + '_normal'
        confusion_matrix_plot(target_trainset_output, estimated_trainset_output.reshape(len(estimated_trainset_output)), confusion_matrix_labels, save_file_name, save_status=True, normalize=True, title='Confusion matrix of train data set')
        #   Test에 대한 confusion matrix 작성
        #save_file_name = save_folder_name + '/kernelsize_' + str(loop_size) + '_kernelnum_' + str(loop_num) + '_test_confusionmatrix'
        #confusion_matrix_plot(target_testset_output, estimated_testset_output.reshape(len(estimated_testset_output)), confusion_matrix_labels, save_file_name, save_status=True, normalize=False, title='Confusion matrix of test data set')
        save_file_name = save_folder_name + '/kernelsize_' + str(loop_size) + '_kernelnum_' + str(loop_num) + '_test_confusionmatrix' + '_normal'
        confusion_matrix_plot(target_testset_output, estimated_testset_output.reshape(len(estimated_testset_output)), confusion_matrix_labels, save_file_name, save_status=True, normalize=True, title='Confusion matrix of test data set')


        '''
        with open(save_file_name_feature, 'at', newline = '') as csvfile :
            writer = csv.writer(csvfile, delimiter = ',')

            writer.writerrow(['Original'] + list())
            for j in range(len(all_featuremaps_names)) :
                writer.writerrow([all_featuremaps_names[j]])
        '''




print('total_accuracy_names 은 : ')
print(total_accuracy_names)
print('total_train_accuracies 은 : ')
print(total_train_accuracies)
print('total_test_accuracies 은 : ')
print(total_test_accuracies)

##  반복을 통해 얻은 train과 test 정확도를 가지고 그래프 확인
fig = plt.figure()
ax = fig.add_subplot(1,1,1) #(행개수, 열개수, 그중 어느것)
ax.plot(total_train_accuracies, 'blue', label='Train accuracy')
ax.plot(total_test_accuracies, 'red', label='Test accuracy')
ax.set_title('Accuracy (Train accuracy(blue) and test accuracy(red))')
#ax.set_xlabel('Kernel size')
ax.set_ylabel('Accuracy')
ax.set_xticks(np.arange(len(total_accuracy_names)))
ax.set_xticklabels(np.arange(1, (len(total_accuracy_names)+1), 1))
#ax.set_xticklabels(total_accuracy_names)
ax.legend(loc='upper left')

save_file_name = save_folder_name + '/loop_accuracy_summary'
save_file_name = save_file_name + '.png'
fig.savefig(save_file_name)
#plt.show()




