import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean

listofarrays = [[[10, 7, 4], [3, 2, 1]], [[12, 8, 5], [4, 3, 2]], [[9, 6, 3], [2, 1, 0]], [[20, 14, 8], [6, 4, 2]]]
array1 = np.array(listofarrays)
medianArray = np.median(array1, axis=0)
meanarray = np.mean(array1, axis=0)
upperquatilearray = np.percentile(array1, q=75, axis=0)
print(medianArray)
print(meanarray)
print(upperquatilearray)
