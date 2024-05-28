import numpy as np
import cv2 as cv


def rotate_point(a, angle):
    x1 = a[0]
    y1 = a[1]

    x2 = np.cos(angle) * x1 - np.sin(angle) * y1
    y2 = np.sin(angle) * x1 - np.cos(angle) * y1

    return np.array([x2, y2])


def rotate_shape(shape, angle):
    result = np.empty((np.shape(shape)[0], 2))
    i = 0
    for vector in shape:
        result[i] = rotate_point(vector, angle)
        i += 1
    return result


batman = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])
print(rotate_shape(batman, np.pi))
