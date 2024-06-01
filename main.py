import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def rotate_point(point, angle):
    x1 = point[0]
    y1 = point[1]

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


def scale_shape(shape, multiplier):
    return shape * multiplier


def mirror_point(point, point_on_line, line_direction):
    return point + 2 * ((np.eye(2, 2) - np.outer(line_direction, np.transpose(line_direction)))
                        @ (point_on_line - point))


def mirror_shape(shape, point_on_line, line_direction):
    result = np.empty((np.shape(shape)[0], 2))
    i = 0
    for vector in shape:
        result[i] = mirror_point(vector, point_on_line, line_direction)
        i += 1
    return result


def shear_shape(shape, shear_matrix):
    result = np.empty((np.shape(shape)[0], 2))
    i = 0
    for vector in shape:
        result[i] = vector @ shear_matrix
        i += 1
    return result


def transform_shape(shape, matrix):
    return shape @ matrix


def draw_shape(shape):
    x = shape[:, :1]
    y = shape[:, 1:]
    plt.plot(x, y)
    plt.show()


star = np.array([[0, 0], [1, 2.5], [-1, 4], [1.5, 4], [2, 6.5], [2.5, 4], [5, 4], [3, 2.5], [4, 0], [2, 2], [0, 0]])
eye = np.array([[3, 0], [0, 2], [1.25, 3.75], [-0.25, 4], [2, 4.75], [0.75, 5], [2.75, 5.5], [1.5, 6], [3.25, 6],
                [5.25, 6.75], [6.5, 6.75], [8.5, 5.75], [9.25, 5], [9.5, 4.25], [9.25, 3.5], [9, 4], [8, 5],
                [8.5, 4], [8.5, 0.75], [6, 0], [8, 1], [8, 2], [7.5, 1.5], [7, 2], [7.5, 2.5], [8, 2], [8, 5],
                [6, 6], [4, 5], [4, 3], [5, 4], [6, 3], [5, 2], [4, 3], [4, 1], [6, 0], [3.5, 0.75], [3.5, 5],
                [6, 6], [3, 5], [2, 4], [1, 2], [3, 0], [8, 0], [9, 0.5], [9, 4]])

draw_shape(eye)
