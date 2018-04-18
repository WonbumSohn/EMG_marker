import itertools
import numpy as np

a = [1,2,3]

b= itertools.product(a, repeat=3)

print([list(np.array(list(b))[0])])