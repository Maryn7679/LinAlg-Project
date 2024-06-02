import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2


def rotate_point(point, angle):
    x1 = point[0]
    y1 = point[1]

    x2 = np.cos(angle) * x1 - np.sin(angle) * y1
    y2 = np.sin(angle) * x1 + np.cos(angle) * y1

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


def shear_shape(shape, x_shear, y_shear):
    result = np.empty((np.shape(shape)[0], 2))
    i = 0
    for vector in shape:
        result[i] = vector @ np.array([[1, x_shear], [y_shear, 1]])
        i += 1
    return result


def transform_shape(shape, matrix):
    return shape @ matrix


def draw_shape(shape):
    x = shape[:, :1]
    y = shape[:, 1:]
    plt.plot(x, y)
    print(shape)
    print()
    plt.show()


def draw_3d_shape(shape):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    x = shape[:, :1]
    y = shape[:, 1:2]
    z = shape[:, 2:]
    ax.plot(x, y, z)
    print(shape)
    print()
    plt.show()


star = np.array([[0, 0], [1, 2.5], [-1, 4], [1.5, 4], [2, 6.5], [2.5, 4], [5, 4], [3, 2.5], [4, 0], [2, 2], [0, 0]])
eye = np.array([[3, 0], [0, 2], [1.25, 3.75], [-0.25, 4], [2, 4.75], [0.75, 5], [2.75, 5.5], [1.5, 6], [3.25, 6],
                [5.25, 6.75], [6.5, 6.75], [8.5, 5.75], [9.25, 5], [9.5, 4.25], [9.25, 3.5], [9, 4], [8, 5],
                [8.5, 4], [8.5, 0.75], [6, 0], [8, 1], [8, 2], [7.5, 1.5], [7, 2], [7.5, 2.5], [8, 2], [8, 5],
                [6, 6], [4, 5], [4, 3], [5, 4], [6, 3], [5, 2], [4, 3], [4, 1], [6, 0], [3.5, 0.75], [3.5, 5],
                [6, 6], [3, 5], [2, 4], [1, 2], [3, 0], [8, 0], [9, 0.5], [9, 4]])

draw_shape(star)
draw_shape(eye)

draw_shape(rotate_shape(eye, np.pi/3))
draw_shape(scale_shape(star, -2))
draw_shape(mirror_shape(eye, np.array([0, 7]), np.array([0, 1])))
draw_shape(shear_shape(star, 7, 3))
draw_shape(transform_shape(star, np.array([[-2, 7], [3, -8]])))
draw_shape(transform_shape(star, np.array([[1, 6], [3, 60]])))
draw_shape(transform_shape(star, np.array([[1, 0], [0, 0]])))

partial_cube = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1],
                         [0, 0, 1], [0, 0, 0]])

draw_3d_shape(partial_cube)

draw_3d_shape(scale_shape(partial_cube, -3))
draw_3d_shape(transform_shape(partial_cube, np.array([[8, 9, 1], [0, 7, 19], [50, 7, -58]])))

image = cv2.imread('woof.jpg')
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

rows, columns = image.shape[:2]
M = cv2.getRotationMatrix2D(((columns - 1) / 2.0, (rows - 1) / 2.0), 60, 1)
result1 = cv2.warpAffine(image, M, (columns, rows))

cv2.imshow('image', result1)
cv2.waitKey(0)
cv2.destroyAllWindows()

rows, columns = image.shape[:2]
pts1 = np.float32([[700, 5], [0, 34], [5, 1100], [641, 529]])
pts2 = pts1 + np.float32([[100, 200], [100, 50], [25, 25], [80, 0]])

M = cv2.getPerspectiveTransform(pts1, pts2)
result2 = cv2.warpPerspective(image, M, (1125, 1105))
plt.subplot(121), plt.imshow(image), plt.title('Input')
plt.subplot(122), plt.imshow(result2), plt.title('Output')
plt.show()
