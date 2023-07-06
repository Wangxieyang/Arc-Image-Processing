import cv2
import scipy.signal as ss
import numpy as np
import os
from PIL import Image


def connect(func, dir):
    imgList = os.listdir('./test0/' + dir)
    # print(imgList)

    sum = 0
    for i in range(0, len(imgList)):
        image = cv2.imread("./test0/" + dir + "/" + imgList[i])
        temp = func(image)

        sum = sum + temp
    average = sum / (len(imgList))
    return average


def area_ms(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (B, G, R) = cv2.split(img)
    gauss = cv2.GaussianBlur(B, (5, 5), 0)
    # print(gauss.shape)
    for i in range(3):
        mid = ss.medfilt2d(gauss, [3, 3])

    _, binary = cv2.threshold(B, 220, 255, cv2.THRESH_BINARY)
    kernal = np.ones((3, 3), np.uint8)

    erosion = cv2.erode(binary, kernal)
    ret0, dio = cv2.threshold(erosion, 0, 255, cv2.THRESH_OTSU)  ##cv2.THRESH_BINARY +

    count1 = count(dio)
    return count1[0]
    # cv2.imshow("img", canny)


def width_ms(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (B, G, R) = cv2.split(img)
    gauss = cv2.GaussianBlur(B, (5, 5), 0)
    # print(gauss.shape)
    for i in range(3):
        mid = ss.medfilt2d(gauss, [3, 3])

    _, binary = cv2.threshold(B, 220, 255, cv2.THRESH_BINARY)
    kernal = np.ones((3, 3), np.uint8)

    erosion = cv2.erode(binary, kernal)
    ret0, dio = cv2.threshold(erosion, 0, 255, cv2.THRESH_OTSU)  ##cv2.THRESH_BINARY +
    canny = cv2.Canny(dio, 100, 200)
    # cv2.imshow("canny",canny)
    # cv2.waitKey()

    temp = np.asarray(canny)
    h = img.shape[0]
    w = img.shape[1]
    minimum = 100
    tmp = 0
    label = 0
    for i in range(215, 315):
        for j in range(w):
            if (temp[i][j] > 0 and label == 0):
                label = 1
            if (temp[i][j] == 0 and label == 1):
                tmp = tmp + 1
            if (temp[i][j] > 0 and label == 1 and tmp > 10):
                minimum = min(minimum, tmp)
                tmp = 0
            else:
                continue
        tmp = 0
        label = 0

    return minimum + 2


def width_draw(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # (B, G, R) = cv2.split(img)
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    # print(gauss.shape)
    for i in range(3):
        mid = ss.medfilt2d(gauss, [3, 3])

    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary", binary)
    kernal = np.ones((3, 3), np.uint8)

    erosion = cv2.erode(binary, kernal)
    ret0, dio = cv2.threshold(erosion, 0, 255, cv2.THRESH_OTSU)  ##cv2.THRESH_BINARY +
    canny = cv2.Canny(dio, 100, 200)
    # cv2.imshow("canny", canny)
    # cv2.waitKey()

    temp = np.asarray(canny)
    h = img.shape[0]
    w = img.shape[1]
    minimum = 50
    itmp0 = 0
    itmp1 = 0
    jtmp0=w-1
    jtmp1=0
    tmp = 0
    label = 0
    for i in range(215, 275):
        for j in range(w):
            if (temp[i][j] > 0):
                if(j<jtmp0):
                    jtmp0=j
                    itmp0=i
                break

    for i in range(215,275):
        for j in range(w-1,0,-1):
            if (temp[i][j] > 0):
                if(j>jtmp1):
                    jtmp1=j
                    itmp1=i
                break






    for i in range(h):
        for j in range(w):
            if (temp[i][j] > 0):
                cv2.circle(img, (j, i), 0, (0, 0, 0), 1)

    dis = (minimum + 2) / 37.67
    dis = round(dis, 2)
    cv2.line(img, (250, itmp0), (500, itmp0), (0, 0, 255), 1)
    cv2.line(img, (250, itmp1), (500, itmp1), (0, 0, 255), 1)
    cv2.putText(img, "w=" + str(dis) + "mm", (250, itmp0 - 10), 0, 0.6, (255, 255, 255), 1, 1)
    cv2.imshow("img", img)

    # 相邻元素提取


def neib(x, y, mat):  # 输入位置和目标数组
    max_x = mat.shape[0] - 1  # 确定行、列最大坐标
    max_y = mat.shape[1] - 1

    matr = []

    # 这里需要重点理解下，坐标不在边缘处，就从前一个遍历到后一个，否则从自身开始
    if x > 0:
        dx = -1
    else:
        dx = 0
    if x < max_x:
        X = 1
    else:
        X = 0
    if y > 0:
        dy = -1
    else:
        dy = 0
    if y < max_y:
        Y = 1
    else:
        Y = 0

    for i in range(dx, X + 1):
        for j in range(dy, Y + 1):
            # if (i == 0 or j == 0) and (i + j != 0):
            if i != 0 or j != 0:
                matr.append(mat[x + i][y + j])
    return matr


# 提取数组最小非零元素

def zxfl(a):
    c = np.asarray(a)
    img5 = c.nonzero()
    imm = c[img5].min()
    return imm


# 标注连续白色区域的像素点

def count(a):
    # 二值化处理

    image = a

    ret1, imageB = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # otsu法取阈值

    temp = np.asarray(imageB)

    X = temp.shape[0]
    Y = temp.shape[1]

    matrix = np.zeros((X, Y))
    matrix = matrix.astype(int)
    label = 1
    # print(matrix)
    for i in range(X):
        for j in range(Y):
            if (temp[i][j] == 255 and matrix[i][j] == 0):
                if max(neib(i, j, matrix)) == 0:
                    matrix[i][j] = label
                    label += 1
                else:
                    matrix[i][j] = zxfl(neib(i, j, matrix))
            else:
                continue

    # print(matrix)
    for i in range(X):
        for j in range(Y):
            if matrix[i][j] != 0:
                if max(neib(i, j, matrix)) != 0 and zxfl(neib(i, j, matrix)) < matrix[i][j]:
                    np.place(matrix, matrix == matrix[i][j], zxfl(neib(i, j, matrix)))

    # print(matrix)
    # np.savetxt("data.txt", matrix)

    S = matrix.max()
    matr = []
    for i in range(S):
        if np.sum(matrix == S) > 0:
            COUNT = np.sum(matrix == S)
            matr.append(COUNT)
        S -= 1

    return matr
