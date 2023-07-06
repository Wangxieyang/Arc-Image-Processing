import cv2
import numpy as np
import matplotlib.pyplot as plt
import function
import func2
import os

if __name__ == '__main__':

    # 注释掉的为文件夹中所有图片批量处理并取平均
    # index = function.connect(func2.width_ms,"10k")
    # print(index)

    # index = function.connect(function.width_ms, "zhi")
    # print(index)
    # index = function.connect(function.area_ms, "20k")
    # print(index)

    image = cv2.imread("./test/10k_00600.jpg")
    cv2.imshow('image',image)
    function.width_draw(image)


    cv2.waitKey()
