import numpy as np
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


batman = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])
print(shear_shape(batman, np.array([[2, 1], [1, 1]])))
print(mirror_shape(batman, np.array([1, 1]), np.array([0, 1])))
