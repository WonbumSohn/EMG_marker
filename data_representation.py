#####---(Learning 이후 train and test data set을 원하는 방법으로 표현하기)---#####

##  설명
#   원하는 deep learning architecture를 통해 learning 한 후 train and test data set을 원하는 방법으로 표현하는 library

##  업데이트 기록지
#   2018.04.17.화요일 : Actual target과 학습된 deep learning architecture를 통해 나온 estimated target에 대한 confusion matrix를 띄우는 함수 추가


#####------------------------------------------------



#####------------------------------------------------

### 사용할 lib 소환
#   반복하는 구문에서 좀 더 다양한 방법으로 방법을 시키기 위해 (for의 응용버전들이라 보면 됨.)
import itertools
#   이미지를 출력하거나 plot을 그리기 위해
import matplotlib.pyplot as plt
#   행렬 연산을 하기 위해
import numpy as np
#   배열 형태로된 confusion matrix를 만들기 위해
from sklearn.metrics import confusion_matrix

#####------------------------------------------------



#####------------------------------------------------

### Confusion matrix를 구하고 원한다면 그래프를 출력or저장하는 함수
##  Normalization can be applied by setting `normalize=True`
def confusion_matrix_plot(actual_target, estimated_target, labels, save_name, save_status=False , normalize=False, title='Confusion matrix', cmap=plt.cm.Blues) :

    ##  변수 초기화

    ##  배열 형태의 confusion matrix를 먼저 생성
    confusion_matrix_array = confusion_matrix(actual_target, estimated_target, labels=labels)

    ##  개수 대신 퍼센트를 보기 위해 normalization 하는 부분
    if normalize:
        #   분모가 0인 경우 그대로 0을 출력하기 위해 'where=__!=0'을 써줌.
        with np.errstate(divide='ignore', invalid='ignore') :
            confusion_matrix_array = confusion_matrix_array.astype('float') / confusion_matrix_array.sum(axis=1)[:, np.newaxis]
            #confusion_matrix_array = np.divide(confusion_matrix_array.astype('float'), confusion_matrix_array.sum(axis=1)[:, np.newaxis], where=(confusion_matrix_array.sum(axis=1)[:, np.newaxis])!=0)
            confusion_matrix_array[np.isnan(confusion_matrix_array)] = 0
            confusion_matrix_array = confusion_matrix_array * 100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    ##  현재의 confusion matrix을 배열 형태로 출력
    print() ; print(title, '는 : ') ; print(confusion_matrix_array)

    ##  Plot하기
    #   새 창 열고
    fig = plt.figure()
    #   Confusion matrix를 그리기
    plt.imshow(confusion_matrix_array, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(ticks = np.arange(0,101,20))
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)#, rotation=45
    plt.yticks(tick_marks, labels)

    #   Normalization을 하면 실수가 되니 '.2f'로 안 하면 정수가 되니 'd'로
    fmt = '.2f' if normalize else 'd'
    #   최대값의 반을 기준으로 아래면 검정글씨, 위면 하얀글씨를 작성하기 위해 최대값의 반을 파악해두는 것
    thresh = confusion_matrix_array.max() / 2.
    #   itertools.product는 ()안에 있는 두 개를 범위 값들을 자동으로 모두 연결하게 반복함
    #   Ex) product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy / product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    for i, j in itertools.product(range(confusion_matrix_array.shape[0]), range(confusion_matrix_array.shape[1])):
        #   숫자 or 퍼센트를 'center(중앙)'에 입력
        plt.text(j, i, format(confusion_matrix_array[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix_array[i, j] > thresh else "black")
    #
    plt.tight_layout()
    #   축의 이름을 배정
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save_status :
        save_file_name = save_name + '.png'
        fig.savefig(save_file_name)
        plt.close(fig)
    else :
        plt.show()

