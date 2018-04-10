import numpy as np

a = [1,2,3,4,5]

print(type(str(a)))
print(str(a))

b = np.array(a) + 10
print(b)

i = 1

if not i == 0 :
    print('i는 0이 아닌 %d 입니다.'%i)