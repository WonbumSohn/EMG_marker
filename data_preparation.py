#####---(Deep Learning에 사용할 data를 불러와 DL에 넣기 전 상태로 가공하기)---#####
##  설명
#   Deep learning에 사용할 data를 여러 형식에 따라 알맞게 불러오고 DL에 넣을 input data와 target(or label) data를 만드는 library

##  업데이트 기록지
#   2018.03.19.월요일 : EMG 4채널 data와 28개의 marker 좌표(x,y,z,az,el,r) data를 불러오기
#                     1개의 csv 파일은 한 피실험자의 한 번의 실험에서 얻은 EMG 4채널 data와 1개의 marker 좌표(x,y,z,az,el,r) data가 들어있음.
#                     열의 이름은 없음
#   2018.03.26.월요일 : Marker 값을 1000개의 구간으로 나눠 label을 만드는 함수와 label을 가지고 다시 값으로 바꾸는(대표값이 되겠지 실제값이 아닌) 함수 완성
#   2018.03.30.금요일 : 최성준의 의견을 고려하여 처음에는 file names만 가져와 조작하고 사용하기로 결정된 최종단계에서 해당하는 files의 data를 불러오는 것으로 수정하여 작동 시간 단축
#   2018.04.04.수요일 : 필요한 files만 로드해오는 부분 최종 완료
#   2018.04.06.금요일 : Target data에서 label data를 만드는 과정, label data에서 target data를 만드는 과정을 딕셔너리를 불러오는 것에서 train target set과 test target set처럼 array를 불러와 처리하는 것으로 변경


#####------------------------------------------------



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
#####------------------------------------------------



#####------------------------------------------------
### 이번 실험의 data들이 들어있는 상위 폴더의 이름을 가지고 해당 폴더에 있는 여러 하위 폴더를 검색하여 모든 data files의 full address를 dictionary 형태로 저장하는 함수
def find_file_address (main_location, folder_name, want_load_type) :

    """

        Input
            main_location       :   모든 data들이 들어있는 폴더의 주소
            folder_name         :   해당 실험의 data가 들어있는 폴더 이름 (주로 실험 제목을 폴더 이름으로 생성)
            want_load_type      :   불어올 data들의 확장자 type   ex) csv

        Output
            files_location      :   모든 csv 파일의 전체 주소가 저장된 dictionary (key는 파일의 이름)
            files_name          :   모든 csv 파일 이름이 저장된 list

    """


    ##  변수 초기화
    #   모든 csv 파일의 주소를 저장할 dictionary 변수 초기화
    files_location = {}
    #   모든 csv 파일의 이름을 저장할 list 변수 초기화
    files_name = []

    ##  해당 실험 data가 들어있는 상위 폴더 주소
    now_data_location = main_location + folder_name + '\\'
    # print(now_data_location)

    ##  해당 폴더에 있는 모든 csv 파일의 이름과 주소를 저장하기 위한 부분 (하위폴더 내에 있는 file까지 모두 저장함)
    #   os.walk :   지정된 폴더의 하위 폴더 안까지 체크해주는 함수
    #   path    :   하위 폴더명 list (맨 처음엔 빈칸이 들어감) ex) D:\WonBumSohn\data\Data_for_LSTM\, D:\WonBumSohn\data\Data_for_LSTM\01, ..., D:\WonBumSohn\data\Data_for_LSTM\28
    #   dir     :   해당 폴더에 존재하는 하위 폴더의 이름을 list로 저장
    #               Ex) path가 'D:\WonBumSohn\data\Data_for_LSTM\' 일 때만 값이 ['01', '02', '03', ..., '27', '28']
    #                   그 외 path에서는 값이 없음
    #   files   :   해당 path 폴더에 들어있는 파일명을 list로 저장 (주소가 들어가는 것이 아니라 단랑 '파일명.확장자'만 들어감.)
    #               Ex) ['cam&emg_01sub_02comb_01mark_trial01.csv', 'cam&emg_01sub_02comb_01mark_trial02.csv', ..., 'cam&emg_21sub_02comb_01mark_trial15.csv']
    for (path, dir, files) in os.walk(now_data_location, topdown=True):
        # print('path : ', path) ; print('dir :', dir) ; print('files : ', files)
        for filename in files:
            #   os.path.splitext()  :   ()안에 입력받은 경로에서 확장자 부분과 그 외 부분으로 나눔. ex) os.path.splitext('C:\\Python30\\python.exe') => ('C:\\Python30\\python', '.exe')
            #   여기서는 files를 가지고 나누는 것이므로 ('cam&emg_01sub_02comb_01mark_trial01', '.csv') 식으로 나뉘고 그중에서 [-1]이므로 '.csv'을 저장하는 것 (불러오길 바라는 파일 확장자명과 실제 불러와 지는 확장자가 같은지 확인하기 위해)
            #   비슷한 기능을 하는 함수
            #   변수.split(구분자)   :   변수 값을 ()안에 있는 구분자를 바탕으로 나눔
            #                           Ex) 'D:\WonBumSohn\data\Data_for_LSTM\01\cam&emg_01sub_02comb_01mark_trial01.csv'.split(\\) => ('D:, WonBumSohn, data, Data_for_LSTM, 01, cam&emg_01sub_02comb_01mark_trial01.csv'
            #   파일의 확장자명 저장
            now_file_type = os.path.splitext(filename)[-1]
            #   파일의 이름을 저장
            now_file_name = os.path.splitext(filename)[0]
            #print(now_file_name)

            #   불러오길 바라는 확장자명과 실제 불러와진 파일의 확장자명이 같을 경우, 다음을 시행 (다를 경우 추후 error로 나타낼 수 있도록 추가할 예정!!!!!!!!!!!!!!!!!!!!)
            if now_file_type == want_load_type:
                #   파일의 이름을 list에 저장
                files_name.append(now_file_name)
                #   해당 파일의 풀 주소
                now_file_location = path + '\\' + filename
                #print(now_file_location)
                #   해당 파일의 전체 주소를 파일이름을 key로 하여 dictionary에 저장
                files_location[now_file_name] = now_file_location

    ##  모든 csv 파일의 전체 주소가 저장된 dictionary와 모든 csv 파일 이름이 저장된 list를 리턴
    ##  files_location = {'cam&emg_01sub_02comb_01mark_trial01' : 'D:\WonBumSohn\data\Data_for_LSTM\01\cam&emg_01sub_02comb_01mark_trial01.csv', ..., 'cam&emg_01sub_02comb_28mark_trial15' : 'D:\WonBumSohn\data\Data_for_LSTM\01\cam&emg_01sub_02comb_28mark_trial15.csv'}
    ##  key의 개수 : 8820=315*28 = list의 길이
    return files_location, files_name



### 이번 deep learning에서 사용할 원하는 markers와 원하는 피실험자들 data 선택하기 위한 함수
def choose_data (files_name, chosen_markers_start, chosen_markers_num, chosen_subjects_start, chosen_subjects_num, subjects_num, trial_num, train_option) :

    """

        Input
            files_name                  :   모든 csv 파일 이름이 저장된 list
            chosen_markers_start        :   사용하고자 하는 markers의 시작점
            chosen_markers_num          :   사용하고자 하는 markers의 개수
            chosen_subjects_start       :   사용하고자 하는 피실험자의 시작점
            chosen_subjects_num         :   사용하고자 하는 피실험자의 인원수
            subjects_num                :   실험에 참여한 전체 피실험자의 인원수
            trial_num                   :   한 피실험자당 반복 측정한 횟수
            train_option                :   Train data를 뽑을 때 어떤 방식으로 추출할지에 대한 옵션 (0 : 모든 피실험자에서 동일한 수의 data를 train data로 추출, 1 : 피실험자를 고려하지 않고 정말 랜덤하게 train data로 추출)
                                            Train data를 뽑는 방식이 다르기 때문에 섞을 때에도 섞는 방식이 다름 (0 : 해당 피실험자 내에서만 섞음, 1 : 전체 피실험자에서 섞음)

        Output
            selected_files_name         :   선택된 files의 name이 저장된 list

    """


    ##  변수 초기화
    #   원하는 마커와 피실험자가 추출된 files의 name을 저장할 list 변수 초기화
    selected_files_name = []

    ##  Train data를 추출하는 방법에 따라 섞는 방법도 다르게 섞음
    if train_option == 'same' : # 모든 피실험자에서 동일한 수의 data를 train data로 추출하기 위해 뽑는 것도 제한적을 뽑아야 함

        ##  사용하고자 하는 markers가 여러 개일 때에는 모든 피실험자를 사용해야 함. (나중에 이것도 선택가능하도록 수정하기!!!!!!!!!!!)
        if chosen_markers_num >= 2 :
            chosen_subjects_start = 1 ; chosen_subjects_num = subjects_num

        #   선택한 구간의 시작점
        start_point = ((chosen_markers_start - 1) * (subjects_num * trial_num)) + ((chosen_subjects_start - 1) * trial_num)
        #   선택한 구간의 끝점
        end_point = (((chosen_markers_start - 1) + (chosen_markers_num - 1)) * (subjects_num * trial_num)) + ((chosen_subjects_start - 1 + chosen_subjects_num) * trial_num)
        #   이번 deep learning에서 사용할 피설험자들 관련 file names 선택
        selected_files_name = files_name[start_point:end_point]

    ##  선택된 files의 name이 저장된 list와 사용되는 marker가 여러 개일 때 사용되는 피실험자 data를 전체로 바꾼 결과를 리턴
    return selected_files_name, [chosen_subjects_start, chosen_subjects_num]



### 불러올 files name이 저장된 list를 이용하여 해당하는 files의 data만 불러오기 위함 함수
def load_data (files_location, files_name, columns_name) :


    '''

        Input
            files_location      :   모든 csv 파일의 전체 주소가 저장된 dictionary (key는 파일의 이름)
            files_name          :   선택한 markers와 피실험자와 관련된 files name만 들어있는 list
            columns_name        :   불러올 파일들의 columns name

        Output
            all_data            :   원하는 files에 들어있는 data가 모두 저장된 dictionary 변수

    '''


    ##  변수 초기화
    #   원하는 files name이 저장한 list를 이용하여 불러온 data를 저장하기 위한 dictionary 변수 초기화
    all_data = {}

    ##  File를 하나씩 불러와 들어있는 data를 불러옴.
    for i in trange(len(files_name)) :
        #   해당 file 안에 든 data를 불러옴.
        now_data = pd.read_csv(files_location[files_name[i]], names=columns_name, encoding='euc-kr')
        #   불러온 data를 array 형태로 변환
        now_data_array = np.array(now_data)
        #   배열로 바뀐 data를 dictionary 변수에 file name을 key로 하여 저장
        all_data[files_name[i]] = now_data_array

        '''
        #   진행중인걸 표시하기 위해
        if (i % 100 == 0) or (i == (len(files_list) - 1)) :
            print('전체 ', len(files_list), '의 files 중에서 ', (i+1), '번째 file을 불러오고 있습니다.')

        #   맨 처음과 맨 마지막 파일의 정보를 출력해서 잘 불러오고 있는지 체크
        if (i == 0) or (i == (len(files_list) - 1)) :
            now_data.info()
            '''

    ##  원하는 files에 들어있는 data가 모두 저장된 dictionary 변수를 리턴
    return  all_data



### 불러온 data set에서 input data와 target data(output의 기준)를 나누기 위한 함수
def separate_data (all_data, files_name, EMG_chs_no, markers_axis_type) :

    '''

        Input
            all_data                    :   원하는 files에 들어있는 data가 모두 저장된 dictionary
            files_name                  :   선택한 markers와 피실험자와 관련된 files name만 들어있는 list
            EMG_chs_no                  :   Data에 들어있는 EMG channel의 개수
            markers_axis_type           :   Markers의 좌표중 어느 좌표값만 target data에 넣을지 결정하는 변수

        Output
            all_input_data              :   Input data set이 저장된 dictionary
            all_target_data             :   Target data set이 저장된 dictionary

    '''


    ##  변수 초기화
    #   Input data set을 저장할 dictionary 변수 초기화
    all_input_data = {}
    #   Target data set을 저장할 dictionary 변수 초기화
    all_target_data = {}

    ##  주어진 EMG 채널과 marker 축 개수에 따라 input data와 target data 나누기
    for i in trange(len(files_name)):

        ##  해당 key 안에 든 파일을 가지고 input data와 target data로 나누기
        #   Input data
        now_input_data = all_data[files_name[i]][:, :EMG_chs_no]

        #   Target data
        now_target_data = all_data[files_name[i]][:, (EMG_chs_no + (markers_axis_type[0] - 1)):(EMG_chs_no + (markers_axis_type[0] - 1) + markers_axis_type[1])]
        # if markers_axis_type == 3 :
        #     now_target_data = all_data[files_name[i]][:, EMG_chs_no:(EMG_chs_no + markers_axis_type)]
        # else :
        #     now_target_data = all_data[files_name[i]][:, (EMG_chs_no + markers_axis_type):(EMG_chs_no + markers_axis_type + 1)]

        ##  나눈 두 개의 data를 각각의 딕셔너리에 저장
        #   Input data
        all_input_data[files_name[i]] = now_input_data
        #   Target data
        all_target_data[files_name[i]] = now_target_data

    ##  나뉜 input data set과 target data set이 저장된 dictionary를 리턴
    return all_input_data, all_target_data



### 원하는 시퀀스 길이에 맞게 input data와 target data를 편집하기 위한 함수
def adjusted_sequence (all_input_data, all_target_data, files_name, seq_len) :

    '''

        Input
            all_input_data              :   Input data set이 저장된 dictionary
            all_target_data             :   Target data set이 저장된 dictionary
            files_name                  :   선택한 markers와 피실험자와 관련된 files name만 들어있는 list
            seq_len                     :   data를 편집할 때 길이를 얼마로 할지 설정하는 변수

        Output
            all_adjusted_input          :   원하는 길이로 편집된 input data set이 저장된 dictionary
            all_adjusted_target         :   원하는 길이로 편집된 target data set이 저장된 dictionary
            all_adjusted_length         :   원하는 길이로 편집된 data들의 길이가 저장된 dictionary

    '''


    ##  변수 초기화
    #   원하는 시퀀스 길이로 편집된 input data set을 저장할 dictionary 변수 초기화
    all_adjusted_input = {}
    #   원하는 시퀀스 길이로 편집된 target data set을 저장할 dictionary 변수 초기화
    all_adjusted_target = {}
    #   원하는 시퀀스 길이로 편집된 data set의 길이를 저장할 dictionary 변수 초기화
    all_adjusted_length = {}

    ##  원하는 시퀀시 길이에 맞게 data 편집하는 부분
    for i in trange(len(files_name)):

        ##  딕셔너리에 key 별로 넣기 위해 한 key에 해당하는 data들을 차근차근 저장하기 위한 변수를 list로 초기화
        now_input_set = []
        now_target_set = []

        #   해당 key 값의 길이
        #   (len(y) - seq_length)이 되면 시퀀스로 뽑혀지는 구간이 한 개가 적으므로 +1 해주어 뽑히는 구간이 끝까지 맞도록 설정
        now_lenght = len(all_target_data[files_name[i]]) - seq_len + 1
        for j in range(0, now_lenght) :
            ##  해당 key 안에 든 data를 가지고 길이 조정
            #   Input data
            #   붙이기 쉽게 잠시 배열을 list로 변환
            now_input_data = list(all_input_data[files_name[i]][j : (j+seq_len)])
            #   List에 합침
            now_input_set.append(now_input_data)
            #   Target data
            #   붙이기 쉽게 잠시 배열을 list로 변환
            now_target_data = list(all_target_data[files_name[i]][(j+seq_len-1)])
            #   List에 합침
            now_target_set.append(now_target_data)

        ##  딕셔너리에 배열 형태로 저장하기 위해 list를 배열로 변환
        now_input_set = np.array(now_input_set)
        now_target_set = np.array(now_target_set)

        ##  나눈 두 개의 data를 각각의 딕셔너리에 저장
        #   Input data
        all_adjusted_input[files_name[i]] = now_input_set
        #   Target data
        all_adjusted_target[files_name[i]] = now_target_set
        #   길이
        all_adjusted_length[files_name[i]] = now_lenght

    ##  원하는 시퀀스 길이 맞게 편집된 input data set, target data set과 편집된 data들의 길이를 files별로 저장한 dictionary들을 리턴
    return all_adjusted_input, all_adjusted_target, all_adjusted_length



# ### Target dat를 classification을 위해 0과 1로 이루어진 label을 생성하기 위한 함수
# def make_num_label(target_data_dic, key_name_list, class_num) :
#
#     ##  Classification을 위해 만든 각 데이터에 대한 label을 딕셔너리에 쭉 저장하기 위해 초기화
#     #   Label 저장용 딕셔너리
#     all_label_dic = {}
#     #   gap 저장용 딕셔너리
#     all_gap_dic = {}
#
#     for i in trange(len(key_name_list)):
#
#         now_target_data = target_data_dic[key_name_list[i]]
#
#         ##  해당 key에 있는 배열의 최대값과 최소값 파악
#         now_max = 1.0 #now_target_data.max()
#         now_min = 0.0 # now_target_data.min()
#         #   원하는 class 수대로 최소~최대 사이를 등분
#         gap = (now_max - now_min) / class_num
#
#         ##  해당 key에 있는 배열의 row size와 column size를 저장
#         row_num = now_target_data.shape[0]
#         #col_num = now_target_data.shape[1]
#
#         ##  변수 초기화
#         #   Classification에서는 해당 class 열 부분에만 1이고 나머지는 0인 label이 필요하므로 처음에는 모든 원소가 0인 행렬을 생성
#         num_label = np.zeros((row_num, class_num))
#         #print('초기화된 숫자 label의 크기는 : ', num_label.shape)
#         #   각 class마다 몇 개의 data가 들어가는지 기록하기 위한 행렬 변수 초기화
#         count_per_label = np.zeros((1, class_num))
#
#         ##  행 하나씩 불러와 크기를 비교하여 label을 만듬.
#         for j in range(row_num):
#
#             ##  원하는 class로 최소-최대 구간을 나눴을 때 해당 행의 값이 어느 구간에 속하는지 판단하여 label 부여
#             for k in range(class_num):
#
#                 ##  원하는 class로 최소-최대 구간을 나눴을 때 해당 행의 값이 어느 구간에 속하는지 판단하여 구간 번째와 동일한 열에 1을 대입
#                 if (now_target_data[j] >= (k * gap)) and (now_target_data[j] < ((k + 1) * gap)):
#                     #   해당하는 열에 1을 대입 (나머지 열은 0)
#                     num_label[j, k] = 1
#                     #   해당하는 class에 속하는 개수를 하나 증가
#                     count_per_label[0, k] = count_per_label[0, k] + 1
#
#             #   1.0보다 큰 값이면 마지막 class에 속하도록 설정
#             if (now_target_data[j] >= now_max):
#                 num_label[j, (class_num - 1)] = 1
#                 #   해당하는 class에 속하는 개수를 하나 증가
#                 count_per_label[0, (class_num - 1)] = count_per_label[0, (class_num - 1)] + 1
#             #   0.0보다 작은 값이면 첫번재 class에 속하도록 설정
#             elif (now_target_data[j] < now_min):
#                 num_label[j, 0] = 1
#                 #   해당하는 class에 속하는 개수를 하나 증가
#                 count_per_label[0, 0] = count_per_label[0, 0] + 1
#
#         ##  각 key에 해당하는 값을 초기화된 딕셔너리에 저장
#         #   Label
#         all_label_dic[key_name_list[i]] = num_label
#         #   Gap
#         all_gap_dic[key_name_list[i]] = gap
#
#
#     print('각 class 별로 들어있는 data의 개수는 : ')
#     for m in range(class_num) :
#         print(str(m+1), '번째 class에 들어있는 data의 개수는 : ', count_per_label[0, m])
#
#
#     ##  Target value에 근거하여 만든 label 딕셔너리와 각 class별 value의 gap을 리턴
#     return all_label_dic, all_gap_dic
#
#
# ### Class를 다시 숫자로 바꾸기 위한 함수
# def return_value (label_data_dic, key_name_list, class_num, all_gap_dic) :
#
#     ##  Classification을 위해 만든 label에 의거하여 다시 대표값으로 바꾼 결과를 딕셔너리에 쭉 저장하기 위해 초기화
#     all_label_value_dic = {}
#
#     for i in trange(len(key_name_list)):
#
#         #   차근차근 key에 든 label을 불러옴.
#         now_key_array = label_data_dic[key_name_list[i]]
#         #   차근차근 key에 든 gap을 불러옴.
#         now_gap = all_gap_dic[key_name_list[i]]
#         #   현재 key에 들은 data의 row size 저장
#         now_key_row_num = now_key_array.shape[0]
#         #   저장한 row size를 바탕으로 class별 대표값을 저장할 행렬 초기화
#         now_key_value = np.zeros((now_key_row_num, 1))
#         #print('초기화된 대표값을 넣을 변수의 크기는 : ', now_key_value.shape)
#
#         ##  행 하나씩 불러와 대표값을 설정
#         for j in range(now_key_row_num):
#
#             #   해당 행에서 1이 들어있는 위치를 저장
#             one_position = np.where(now_key_array[j] == 1)[0]
#
#             #   위치에 따라 다른 대표값을 저장
#             for k in range(class_num):
#
#                 if one_position == k:
#                     #   해당 행에 1이 있는 위치에 최소~최대 사이를 class의 수로 나눈 gap 만큼을 곱해 대표값으로 설정
#                     now_key_value[j] = k * now_gap
#
#         all_label_value_dic[key_name_list[i]] = now_key_value
#
#     ##  Label을 근거로 다시 값(대표값)으로 바꾼 결과 딕셔너리를 리턴
#     return all_label_value_dic



### Data를 랜덤하게 섞는 효과를 주기 위해 딕셔너리 key의 정보가 담긴 files의 이름이 담긴 list를 불러와 랜덤하게 섞음.
def shuffle_data (files_name, chosen_markers_num, chosen_subjects_num, trial_num, train_option, subjects_num) :

    '''

        Input
            files_name              :   사용하고자 하는 marker와 피실험자 조건에 따라 선택된 files의 name이 저장된 list
            chosen_markers_num      :   사용하고자 하는 markers의 개수
            chosen_subjects_num     :   사용하고자 하는 피실험자의 인원수
            trial_num               :   한 피실험자당 반복 측정한 횟수
            train_option            :   Train data를 뽑을 때 어떤 방식으로 추출할지에 대한 옵션 (0 : 모든 피실험자에서 동일한 수의 data를 train data로 추출, 1 : 피실험자를 고려하지 않고 정말 랜덤하게 train data로 추출)
                                        Train data를 뽑는 방식이 다르기 때문에 섞을 때에도 섞는 방식이 다름 (0 : 해당 피실험자 내에서만 섞음, 1 : 전체 피실험자에서 섞음)
            subjects_num            :   실험에 참여한 전체 피실험자의 인원수

        Output
            shuffled_files_name     :   원하는 조건으로 랜덤하게 섞은 files의 name이 저장된 list

    '''

    ##  File 이름을 랜덤하게 섞은 결과를 저장하기 위한 list 변수
    shuffled_files_name = []

    ##  Train data를 추출하는 방법에 따라 섞는 방법도 다르게 섞음
    #   모든 피실험자에서 동일한 수의 data를 train data로 추출하기 위해 files의 name을 섞는 것도 한 피실험자 내에서 섞음.
    #if train_option == 0 :
    if train_option == 'same' :

        ##  사용하고자 하는 markers가 여러 개일 때에는 모든 피실험자를 사용해야 함. (나중에 이것도 선택가능하도록 수정하기!!!!!!!!!!!)
        #if chosen_markers_num >= 2:
        #    chosen_subjects_num = subjects_num

        #   한 개의 피실험자당 마커가 여러개 있으므로 구간을 아래와 같이 설정 ex) 21명, 28개 마커, 15번 반복 ==> 15번의 반복이 21*28개 있는 것
        for i in trange((chosen_markers_num * chosen_subjects_num)) :
            #   랜덤하게 섞을 구간을 선택
            chosen_file_name = files_name[(i * trial_num):((i + 1) * trial_num)]
            #   랜덤하게 섞기
            random.shuffle(chosen_file_name)
            #   랜덤하게 섞은 것을 한 개의 변수에 합치기
            shuffled_files_name = shuffled_files_name + chosen_file_name

    ##  원하는 조건으로 랜덤하게 섞은 files의 name이 저장된 list를 리턴
    return shuffled_files_name



### 원하는 train 비율만큼 train set을 만들고 나머지는 test set을 만들기 위해 train 비율만큼 선택된 file name 중 일부를 train set으로 지정 후 나머지는 test set으로 지정하기 위한 함수
def make_train_test_list_set (files_name, trial_num, train_rate, chosen_markers_num, chosen_subjects_num, train_option) :

    '''

        Input
            files_name              :   사용하고자 하는 marker와 피실험자 조건에 따라 선택된 files의 name이 저장된 list
            trial_num               :   한 피실험자당 반복 측정한 횟수
            train_rate              :   불러온 data 중에서 몇 퍼센트를 train data로 쓸지에 대한 변수
            chosen_markers_num      :   사용하고자 하는 markers의 개수
            chosen_subjects_num     :   사용하고자 하는 피실험자의 인원수
            train_option

        Output
            train_files             :   정해진 비율에 따라 train data로 지정된 files의 이름이 저장된 list
            test_files              :   정해진 비율에 따라 test data로 지정된 files의 이름이 저장된 list

    '''


    ##  변수 초기화
    #   전체 train files name를 저장할 list 변수 초기화
    train_files = []
    #   전체 test files name를 저장할 list 변수 초기화
    test_files = []

    ##  train option에서 모든 피실험자에서 동일한 수의 data를 train data로 추출하는 것을 선택했을 때
    if train_option == 'same' : # 모든 피실험자에서 동일한 수의 data를 train data로 추출
        #   입력한 비율만큼을 이용하여 train set을 모든 피실험자에서 균일하게 추출하기 위해 반복 횟수에 train 비율을 곱해 train size를 설정
        train_size = int(trial_num * train_rate)
        print('각 피실험자당 train data size는 : ', train_size)
        #   반복 횟수에서 train size를 뺀 값을 test size로 설정
        test_size = trial_num - train_size
        print('각 피실험자당 test data size는 : ', test_size)
        print('train_size ', train_size, '+ test_size ', test_size, ' = 한 피실험자당 반복 측정 횟수 ', trial_num)
        print('위의 수식이 성립해면 제대로 구한 것')

        ##  정해진 train size와 test size를 바탕으로 files name를 train set과 test set으로 나누기
        for i in trange(chosen_markers_num * chosen_subjects_num) :
            #   한 피실험자 내에서 train용 files name을 추출
            now_train_name = files_name[(i * trial_num) : ((i * trial_num) + train_size)]
            #   train 전체 files name을 저장할 list에 추가
            train_files = train_files + now_train_name

            #   한 피실험자 내에서 test용 files name을 추출
            now_test_name = files_name[((i * trial_num) + train_size): ((i + 1) * trial_num)]
            #   test 전체 files name을 저장할 list에 추가
            test_files = test_files + now_test_name

    ##  정해진 비율에 따라 만들어진 train용 files name와 test용 files name를 저장한 list를 리턴
    return train_files, test_files



### 위에서 train용과 test용 file names를 가지고 실제 data와 target도 train용과 test용으로 나누기 위한 함수(행렬의 행결합은 np.r_())
def make_train_test_set (all_adjusted_input, all_adjusted_target, train_files, test_files) :

    """

        Input
            all_adjusted_input      :   원하는 길이로 편집된 input data set이 저장된 dictionary
            all_adjusted_target     :   원하는 길이로 편집된 target data set이 저장된 dictionary
            train_files             :   정해진 비율에 따라 train data로 지정된 files의 이름이 저장된 list
            test_files              :   정해진 비율에 따라 test data로 지정된 files의 이름이 저장된 list

        Output
            train_data              :   Train input data의 값이 저장된 array
            train_target            :   Train target data의 값이 저장된 array
            test_data               :   Test input data의 값이 저장된 array
            test_target             :   Test target data의 값이 저장된 array

    """


    ##  변수 초기화
    #   Train input data를 저장할 list 변수 초기화 (나중에 array로 바꿔줌)
    train_data = []
    #   Train target data를 저장할 list 변수 초기화 (나중에 array로 바꿔줌)
    train_target = []
    #   Test input data를 저장할 list 변수 초기화 (나중에 array로 바꿔줌)
    test_data = []
    #   Test target data를 저장할 list 변수 초기화 (나중에 array로 바꿔줌)
    test_target = []

    ##  Train data와 train target 만드는 부분
    print('Train data set을 만들고 있습니다.') ; print()
    for i in trange(len(train_files)) :
        #   Train names list에 든 name을 하나씩 불러와 거기에 해당하는 data와 target를 별도로 추출
        #   List로 해주어야 단순히 +로 행결합을 할 수 있어 list화
        now_data = list(all_adjusted_input[train_files[i]])
        now_target = list(all_adjusted_target[train_files[i]])
        #   List의 행결합
        train_data = train_data + now_data
        train_target = train_target + now_target

    #   Deep learning architecture의 입력을 위해 다시 배열로 변환
    train_data = np.array(train_data)
    train_target = np.array(train_target)

    ##  Test data와 test target 만드는 부분
    print('Test data set을 만들고 있습니다.') ; print()
    for i in trange(len(test_files)) :
        #   Train names list에 든 name을 하나씩 불러와 거기에 해당하는 data와 target를 별도로 추출
        #   List로 해주어야 단순히 +로 행결합을 할 수 있어 list화
        now_data = list(all_adjusted_input[test_files[i]])
        now_target = list(all_adjusted_target[test_files[i]])
        #   List의 행결합
        test_data = test_data + now_data
        test_target = test_target + now_target

    #   Deep learning architecture의 입력을 위해 다시 배열로 변환
    test_data = np.array(test_data)
    test_target = np.array(test_target)

    ##  나눠진 train data set, train target set, test data set, test target set을 리턴
    return train_data, train_target, test_data, test_target



### Target dat를 classification을 위해 0과 1로 이루어진 label을 생성하기 위한 함수
def make_num_label(train_target, test_target, class_num) :

    '''

        Input
            train_target            :   Train target data의 값이 저장된 array
            test_target             :   Test target data의 값이 저장된 array
            class_num               :   Classification을 위해 target data를 몇 개의 class로 나눌지 class 개수를 지정

        Output
            train_num_label         :   Train target data로 부터 구해진 train label이 저장된 array
            test_num_label          :   Test target data로 부터 구해진 Test label이 저장된 array
            train_target_min        :   Train target data의 최소값
            train_target_max        :   Train target data의 최대값
            train_gap_value         :   Train target data의 최소, 최대, class의 개수를 바탕으로 구해진 각 class마다 target value의 최초-최대 차이

    '''


    ##  변수 초기화
    #   Train target data의 최소, 최대값을 지정
    train_target_min = np.nanmin(train_target, axis=0) ; train_target_max = np.nanmax(train_target, axis=0)
    #   Train target의 최소, 최대와 원하는 class 수를 고려하여 계산한 gap 값 (최소~최대 사이를 class수 대로 N 등분)
    #   Test target이 train target의 최소, 최대를 벗어나는 경우를 생각하여 class 구간은 train target을 이용하여 구하는 것)
    train_gap_value = (train_target_max - train_target_min) / class_num

    #   Train target과 test target의 row size를 저장
    tarin_row_num = train_target.shape[0] ; test_row_num = test_target.shape[0]

    #   Classification에서는 해당 class 열 부분에만 1이고 나머지는 0인 label이 필요하므로 처음에는 모든 원소가 0인 행렬을 생성
    train_num_label = np.zeros((tarin_row_num, class_num)) ; test_num_label = np.zeros((test_row_num, class_num))
    #print('Train target의 초기화된 숫자 label의 크기는 : ', train_num_label) ; print('Test target의 초기화된 숫자 label의 크기는 : ', test_num_label)
    #   각 class마다 몇 개의 data가 들어가는지 기록하기 위한 행렬 변수 초기화
    train_count_per_label = np.zeros((1, class_num)) ; test_count_per_label = np.zeros((1, class_num))

    ##  행 하나씩 불러와 크기를 비교하여 train label을 만듬.
    for i in trange(tarin_row_num):

        ##  원하는 class로 최소-최대 구간을 나눴을 때 해당 행의 값이 어느 구간에 속하는지 판단하여 label 부여
        for j in range(class_num):

            ##  원하는 class로 최소-최대 구간을 나눴을 때 해당 행의 값이 어느 구간에 속하는지 판단하여 구간 번째와 동일한 열에 1을 대입
            if (train_target[i] >= (train_target_min + (j * train_gap_value))) and (train_target[i] < (train_target_min + ((j + 1) * train_gap_value))):
                #   해당하는 열에 1을 대입 (나머지 열은 0)
                train_num_label[i, j] = 1
                #   해당하는 class에 속하는 개수를 하나 증가
                train_count_per_label[0, j] = train_count_per_label[0, j] + 1

        #   최소값보다 작은 값이면 첫번재 class에 속하도록 설정
        if (train_target[i] < train_target_min):
            train_num_label[i, 0] = 1
            #   해당하는 class에 속하는 개수를 하나 증가
            train_count_per_label[0, 0] = train_count_per_label[0, 0] + 1
        #   최대값보다 큰 값이면 마지막 class에 속하도록 설정
        elif (train_target[i] >= train_target_max):
            train_num_label[i, (class_num - 1)] = 1
            #   해당하는 class에 속하는 개수를 하나 증가
            train_count_per_label[0, (class_num - 1)] = train_count_per_label[0, (class_num - 1)] + 1

    ##  행 하나씩 불러와 크기를 비교하여 test label을 만듬.
    for i in trange(test_row_num):

        ##  원하는 class로 최소-최대 구간을 나눴을 때 해당 행의 값이 어느 구간에 속하는지 판단하여 label 부여
        for j in range(class_num):

            ##  원하는 class로 최소-최대 구간을 나눴을 때 해당 행의 값이 어느 구간에 속하는지 판단하여 구간 번째와 동일한 열에 1을 대입
            if (test_target[i] >= (train_target_min + (j * train_gap_value))) and (test_target[i] < (train_target_min + ((j + 1) * train_gap_value))):
                #   해당하는 열에 1을 대입 (나머지 열은 0)
                test_num_label[i, j] = 1
                #   해당하는 class에 속하는 개수를 하나 증가
                test_count_per_label[0, j] = test_count_per_label[0, j] + 1

        #   최소값보다 작은 값이면 첫번재 class에 속하도록 설정
        if (test_target[i] < train_target_min):
            test_num_label[i, 0] = 1
            #   해당하는 class에 속하는 개수를 하나 증가
            test_count_per_label[0, 0] = test_count_per_label[0, 0] + 1
        #   최대값보다 큰 값이면 마지막 class에 속하도록 설정
        elif (test_target[i] >= train_target_max):
            test_num_label[i, (class_num - 1)] = 1
            #   해당하는 class에 속하는 개수를 하나 증가
            test_count_per_label[0, (class_num - 1)] = test_count_per_label[0, (class_num - 1)] + 1


    ##  각 class 별로 들어있는 data의 개수 확인
    print('각 class 별로 들어있는 train data의 개수는 : ')
    for m in range(class_num) :
        print('     ', str(m+1), '번째 class에 들어있는 train data의 개수는 : ', train_count_per_label[0, m])
    print('각 class 별로 들어있는 test data의 개수는 : ')
    for m in range(class_num):
        print('     ', str(m + 1), '번째 class에 들어있는 test data의 개수는 : ', test_count_per_label[0, m])


    ##  Train과 test target value에 근거하여 만든 train과 test label array와 train target의 최소, 최대, class 개수를 고려하여 계산한 gap 값을 리턴
    return train_num_label, test_num_label, train_target_min, train_target_max, train_gap_value



### Class를 다시 숫자로 바꾸기 위한 함수 (estimated를 이용할 때)
def return_value (num_label, class_num, train_target_min, train_gap_value) :

    '''

        Input
            num_label               :   Target data로 부터 구해진 label이 저장된 array (Train label이거나 test label임)
            class_num               :   Classification을 위해 target data를 몇 개의 class로 나눌지 class 개수를 지정
            train_target_min        :   Train target data의 최소값
            train_gap_value         :   Train target data의 최소, 최대, class의 개수를 바탕으로 구해진 각 class마다 target value의 최초-최대 차이

        Output
            estimated_value         :   Label을 근거로 다시 값(대표값:각 그룹의 범위의 중간값)으로 바꾼 결과를 저장한 array

    '''


    ##  초기화
    #   Label data의 row size 저장
    row_num = num_label.shape[0]
    #   Classification을 위해 만든 label에 의거하여 다시 대표값으로 바꾼 결과를 배열로 쭉 저장하기 위해 class별 대표값을 저장할 array 초기화 (제로행렬)
    estimated_value = np.zeros((row_num, 1))
    #print('초기화된 대표값을 넣을 변수의 크기는 : ', now_key_value.shape)

    ##  행 하나씩 불러와 대표값을 설정
    for i in trange(row_num):

        #   행의 값에 따라 다른 대표값을 저장
        for j in range(class_num):

            if num_label[i] == j:
                #   해당 행에 1이 있는 위치에 최소~최대 사이를 class의 수로 나눈 gap 만큼을 곱해 대표값으로 설정
                estimated_value[i] = train_target_min + (j * train_gap_value) + (train_gap_value/2)

    ##  Label을 근거로 다시 값(대표값:각 그룹의 범위의 중간값)으로 바꾼 결과를 저장한 array를 리턴
    return estimated_value



# ### Class를 다시 숫자로 바꾸기 위한 함수 (라벨 제작중 바로 확인용)
# def return_value (num_label, class_num, train_target_min, train_gap_value) :
#
#     '''
#
#         Input
#             num_label               :   Target data로 부터 구해진 label이 저장된 array (Train label이거나 test label임)
#             class_num               :   Classification을 위해 target data를 몇 개의 class로 나눌지 class 개수를 지정
#             train_target_min        :   Train target data의 최소값
#             train_gap_value         :   Train target data의 최소, 최대, class의 개수를 바탕으로 구해진 각 class마다 target value의 최초-최대 차이
#
#         Output
#             estimated_value         :   Label을 근거로 다시 값(대표값:각 그룹의 범위의 중간값)으로 바꾼 결과를 저장한 array
#
#     '''
#
#
#     ##  초기화
#     #   Label data의 row size 저장
#     row_num = num_label.shape[0]
#     #   Classification을 위해 만든 label에 의거하여 다시 대표값으로 바꾼 결과를 배열로 쭉 저장하기 위해 class별 대표값을 저장할 array 초기화 (제로행렬)
#     estimated_value = np.zeros((row_num, 1))
#     #print('초기화된 대표값을 넣을 변수의 크기는 : ', now_key_value.shape)
#
#     ##  행 하나씩 불러와 대표값을 설정
#     for i in trange(row_num):
#
#         #   해당 행에서 1이 들어있는 위치를 저장
#         one_position = np.where(num_label[i] == 1)[0]
#
#         #   위치에 따라 다른 대표값을 저장
#         for j in range(class_num):
#
#             if one_position == j:
#                 #   해당 행에 1이 있는 위치에 최소~최대 사이를 class의 수로 나눈 gap 만큼을 곱해 대표값으로 설정
#                 estimated_value[i] = train_target_min + (j * train_gap_value) + (train_gap_value/2)
#
#     ##  Label을 근거로 다시 값(대표값:각 그룹의 범위의 중간값)으로 바꾼 결과를 저장한 array를 리턴
#     return estimated_value




















