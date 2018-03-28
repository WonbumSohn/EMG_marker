#####---(Deep Learning을 이용한 프로그램)---#####
##  설명
#   Deep learning에 사용하여 원하는 알고리즘을 통해 원하는 타입의 학습을 진행하는 main 부분

##  업데이트 기록지
#   2018.03.19.월요일 : EMG 4채널 data와 28개의 marker 좌표(x,y,z,az,el,r) data를 불러와
#                      EMG 4채널 data로 부터 우선 1개의 marker 좌표(x,y,z,az,el,r)를 regression하는 알고리즘 구현 시작
#   2018.03.26.월요일 : Marker 값을 1000개의 구간으로 나눠 label을 만드는 것까지 완성
#                      다시 값(대표값)으로 바꿨을 때 거의 차이가 없었음.
#   2018.03.28.수요일 : GPU설치로 tensorflow를 gpu로 돌리는 것으로 체인지!

#####------------------------------------------------


### 사용할 lib 소환
from data_preparation import *

# 이미지를 출력하거나 plot을 그리기 위해
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

### 사용할 고정 변수 입력
##  Deep learning을 위한 data를 준비하면서 필요한 변수들
#   불러올 파일이 든 폴더 이름
folder_name = 'Data_for_LSTM'
#   불러올 파일의 확장자명
want_file_type = '.csv'
#   불러올 파일들의 열의 이름
data_column_names = ['EMG_ch1', 'EMG_ch2', 'EMG_ch3', 'EMG_ch4', 'marker_x', 'marker_y', 'marker_z', 'marker_az', 'marker_el', 'marker_r']

#   EMG channel의 개수
EMG_ch_num = 4
#   Marker의 좌표 개수
marker_axis_num = 3
#   실제 사용할 marker 좌표 (0 : x축, 1 : y축, 2 : z축, 3 : x,y,z 모두)
wanted_marker = 0

#   1개 시퀀스의 길이 (시계열데이터 입력 개수)
seq_length = 20

#   Marker의 개수
marker_num = 28
#   피실험자 수
subject_num = 21
#   실험 반복 횟수
trial_num = 15
#   Train data를 뽑는 옵션 (0 : 모든 피실험자에서 동일한 수의 data를 train data로 추출, 1 : 피실험자를 고려하지 않고 정말 랜덤하게 train data로 추출)
train_option = 0

#   전체 markers 중에서 몇 개의 markers를 !!연달아!! 선택할지와 무슨 makrer부터 선택할지([marker 개수, marker 선택 시작점])
chosen_marker_num = [1, 1]
#   전체 피실험자 중 이번 deep learning에 사용할 피실험자 수와 어떤 피실험자부터 !!연달아!! 선택할지([피실험자 수, 피실험자 선택 시작점]) (marker 개수가 !!여러개!!면 자동으로 [subject_num , 1]
chosen_subject_num = [1, 1]

#   얼마만큼을 train set으로 쓸지 비율 입력
train_ratio = 0.7

#   Classification을 위해 나누고자 하는 class 수
class_num = 1000


##  CNN을 돌리기 위한 변수들





### Data 불러오기
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100)
print('Data를 불러오는 부분')
#   지정된 폴더 안에 있는 모든 하위 폴더에 든 data files의 주소를 한 번에 list 형태로 저장
files_list = find_files(folder_name, want_file_type)
print('지정된 폴더 안에 있는 모든 data files의 개수는(A) : ', len(files_list))
#   저장한 files의 위치들을 바탕으로 data와 file 이름을 불러오기
total_data, total_file_names = load_data(files_list, want_file_type, data_column_names)
print('전체 data의 개수는(A와 같아야 함) : ', len(total_data))
print('전체 data의 key 이름 개수는(A와 같아야 함) : ', len(total_file_names))
#print('전체 data는 : ')
#print(total_data)
print('File들의 이름은(딕셔너리 key명) : ')
print(total_file_names)
print(total_file_names[0], ' 파일 안에 든 data의 측정 길이는(B) ', len(total_data[total_file_names[0]]))
print(total_file_names[0], ' 파일 안에 든 data의 tpye은 ', type(total_data[total_file_names[0]]))


### Data에서 input data와 output의 기준이 되는 target data를 나눔. (이때 몇 개의 축을 선택할지 고름)
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100)
print('불러온 파일 data에서 input data와 target data로 나누는 부분')
input_data_set, target_data_set = separate_data (total_data, total_file_names, EMG_ch_num, wanted_marker)
print('Input data의 총 개수는(A와 같아야 함) : ', len(input_data_set))
print('Target data의 총 개수는(A와 같아야 함) : ', len(target_data_set))
print(total_file_names[0], ' 파일 안에 든 input data의 측정 길이는(B와 같아야 함) : ', (input_data_set[total_file_names[0]]).shape)
print(total_file_names[0], ' 파일 안에 든 input data의 tpye은 : ', type(input_data_set[total_file_names[0]]))
print(total_file_names[0], ' 파일 안에 든 input data는 : ')
print(input_data_set[total_file_names[0]])
print(total_file_names[0], ' 파일 안에 든 target data의 측정 길이는(B와 같아야 함) : ', (target_data_set[total_file_names[0]]).shape)
print(total_file_names[0], ' 파일 안에 든 target data의 tpye은 : ', type(target_data_set[total_file_names[0]]))
print(total_file_names[0], ' 파일 안에 든 target data는 : ')
print(target_data_set[total_file_names[0]])


### 한 마커당 한 피실험자의 반복 실험을 통해 얻은 data의 순서를 랜덤하게 섞음.
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100)
print('한 마커당 한 피실험자의 반복 실험을 통해 얻은 data의 순서를 랜덤하게 섞는 부분')
##  File(딕셔너리 key)의 이름을 저장해놓은 list만 섞으면 딕셔너리를 자동으로 섞이는 효과 발생
re_file_names = shuffle_data (total_file_names, marker_num, subject_num, trial_num, train_option)
print('섞인 후 file names의 개수는(A와 같아야 함) : ', len(re_file_names))
print('섞인 후 file names는 : ')
print(re_file_names)
print('섞인 후 file names을 체크')
print('첫번째 마커, 첫번째 피실험자 : ')
print(re_file_names[0:15])
print('첫번째 마커, 두번째 피실험자 : ')
print(re_file_names[15:30])


### 이번 deep learning에서 사용할 피실험자들 data 선택
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100)
print('이번 deep learning에서 사용할 피실험자들 data를 선택하는 부분')
chosen_file_names = choose_data (re_file_names, chosen_marker_num, chosen_subject_num, subject_num, trial_num, train_option)
print('선택한 data의 file names은 : ')
print(chosen_file_names)


### 원하는 시퀀스 길이에 맞게 data 편집
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100)
print('원하는 시퀀스 길이에 맞게 data를 편집')
adjusted_input_set, adjusted_target_set, adjusted_length = adjusted_sequence (input_data_set, target_data_set, chosen_file_names, seq_length)
print('원하는 시퀀스 길이로 편집된 input data의 총 개수는(A와 같아야 함) : ', len(adjusted_input_set))
print('원하는 시퀀스 길이로 편집된 target data의 총 개수는(A와 같아야 함) : ', len(adjusted_target_set))
print('원하는 시퀀스 길이로 편집된 length의 총 개수(길이)는(A와 같아야 함) : ', len(adjusted_length))
print(chosen_file_names[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 input data의 길이는(C, 한 실험에 대한 RNN에 input으로 들어갈 data 개수) : ', len(adjusted_input_set[chosen_file_names[0]]))
print(chosen_file_names[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 input data의 tpye은 : ', type(adjusted_input_set[chosen_file_names[0]]))
print(chosen_file_names[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 input data의 shape은 : ', (adjusted_input_set[chosen_file_names[0]]).shape)
print(chosen_file_names[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 input data는 : ')
print(adjusted_input_set[chosen_file_names[0]])
print(chosen_file_names[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 target data의 길이는(C와 같아야 함) : ', len(adjusted_target_set[chosen_file_names[0]]))
print(chosen_file_names[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 target data의 tpye은 : ', type(adjusted_target_set[chosen_file_names[0]]))
print(chosen_file_names[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 target data의 shape은 : ', (adjusted_target_set[chosen_file_names[0]]).shape)
print(chosen_file_names[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 target data는 : ')
print(adjusted_target_set[chosen_file_names[0]])
print(chosen_file_names[0], ' 파일 안에 든 원하는 시퀀스 길이로 편집된 input data의 길이는(C와 같아야 함) : ', (adjusted_length[chosen_file_names[0]]))
print('편집된 data들 중 1번째 마커, 1번재 피실험자의 data들 길이는 : ')
for i in range(15) :
    print(adjusted_length[chosen_file_names[i]])


### Target을 classification의 label로 만드는 부분 (Classification을 위한 부분)
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100)
print('Classification을 위해 label을 만드는 중')
all_label, all_gap = make_num_label(adjusted_target_set, chosen_file_names, class_num)
print('Classification을 위해 label로 바꾼 딕셔너리의 총 key 개수는(A와 같아야 함) : ', len(all_label))
print(chosen_file_names[0], ' 라는 key 안에 든 실제값의 간격은 : ', all_gap[chosen_file_names[0]])
print(chosen_file_names[0], ' 라는 key 안에 든 label의 길이는(C, 한 실험에 대한 RNN에 input으로 들어갈 data 개수) : ', len(all_label[chosen_file_names[0]]))
print(chosen_file_names[0], ' 라는 key 안에 든 label의 tpye은 : ', type(all_label[chosen_file_names[0]]))
print(chosen_file_names[0], ' 라는 key 안에 든 label의 shape은 : ', (all_label[chosen_file_names[0]]).shape)
print(chosen_file_names[0], ' 라는 key 안에 든 label은 : ')
print(adjusted_target_set[chosen_file_names[0]][0])
print(all_label[chosen_file_names[0]][0])

'''
### Class를 다시 숫자로 바꾸는 부분
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100)
print('그래프를 그리기 위해 label을 다신 대표값으로 바꾸는 중')
all_label_value = return_value (all_label, chosen_file_names, class_num, all_gap)
print('Label에 의거하여 대표값을 넣은 딕셔너리의 총 key 개수는(A와 같아야 함) : ', len(all_label_value))
print(chosen_file_names[0], ' 라는 key 안에 든 대표값 배열의 길이는(C, 한 실험에 대한 RNN에 input으로 들어갈 data 개수) : ', len(all_label_value[chosen_file_names[0]]))
print(chosen_file_names[0], ' 라는 key 안에 든 대표값 배열의 tpye은 : ', type(all_label_value[chosen_file_names[0]]))
print(chosen_file_names[0], ' 라는 key 안에 든 대표값 배열의 shape은 : ', (all_label_value[chosen_file_names[0]]).shape)
print(chosen_file_names[0], ' 라는 key 안에 든 대표값 배열은 : ')
print(adjusted_target_set[chosen_file_names[0]][0:10])
print(all_label_value[chosen_file_names[0]][0:10])


##  원래 값과 label을 다시 값으로 바꾼 대표값 사이의 차이 정도를 보기 위해 그래프 출력
plt.figure(1)
plt.plot(adjusted_target_set[chosen_file_names[0]], 'blue')
plt.plot(all_label_value[chosen_file_names[0]], 'red')
plt.show()
'''

### 원하는 train 비율만큼 train set을 만들고 나머지는 test set을 만들기 위해 train 비율만큼 선택된 file name 중 일부를 train set으로 지정 후 나머지는 test set으로 지정
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100)
print('원하는 train 비율만큼 train set용 data 이름과 test set용 data 이름을 만드는 부분')
train_file_names, test_file_names = make_train_test_list_set (chosen_file_names, trial_num, train_ratio, chosen_marker_num, chosen_subject_num, train_option)
print('Train용 file names의 개수는 : ', len(train_file_names))
print('Train용 file names 는 : ')
print(train_file_names)
print('Test용 file names의 개수는 : ', len(test_file_names))
print('Test용 file names 는 : ')
print(test_file_names)


### 위에서 train용과 test용 file names를 가지고 실제 data와 target도 train용과 test용으로 나누기
#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100)
print('위에서 나눈 data 이름을 바탕으로 실제 train set data와 test set data를 만드는 부분')
train_data_set, train_label_set, test_data_set, test_label_set = make_train_test_set(adjusted_input_set, all_label, train_file_names, test_file_names)
#train_data_set, train_target_set, test_data_set, test_target_set = make_train_test_set(adjusted_input_set, adjusted_target_set, train_file_names, test_file_names)
print('Train data의 type은 : ', type(train_data_set))
print('Train data의 크기는 : ', (train_data_set).shape)
print('Train data는 : ')
print(train_data_set)
print('Train target의 type은 : ', type(train_label_set))
print('Train target의 크기는 : ', (train_label_set).shape)
print('Train target은 : ')
print(train_label_set[0])
print('Test data의 type은 : ', type(test_data_set))
print('Test data의 크기는 : ', (test_data_set).shape)
print('Test data는 : ')
print(test_data_set)
print('Test target의 type은 : ', type(test_label_set))
print('Test target의 크기는 : ', (test_label_set).shape)
print('Test target은 : ')
print(test_label_set[0])


#   화면상 구분을 위해 ('=======================' 이걸 긋는 효과)
print('=' * 100)
print('=' * 100)
print('이제부터는 본격적으로 CNN training에 들어갑니다.')
print('=' * 100)
print('=' * 100)








