import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn import svm

def iqos_or_cig (path):
    im = cv.imread(path, 0)
    #iqos_rgb = cv.imread(path)
    #ret, im2 = cv.threshold(im, 170, 255, cv.THRESH_BINARY)  # Binarizing image
    ret, im2 = cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    # closing=cv.morphologyEx(dst, cv.MORPH_CLOSE, kernel) #Closing operation
    # sobel = cv.Sobel(iqos,cv.CV_8U,1,0,ksize=5)
    # plt.imshow(sobel)
    _, cnt, hierarchy = cv.findContours(im2, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)  # Getting contours
    if len(cnt) != 0:
        rect = cv.minAreaRect(max(cnt, key = cv.contourArea))
        #box = cv.boxPoints(rect)
        #box = np.int0(box)
        #plt.imshow(cv.drawContours(iqos_rgb, [box], 0, (0, 0, 0), 1), cmap='gray')
        width=rect[1][0]
        height=rect[1][1]
        threshold = 3.5

        area = cv.contourArea(max(cnt, key=cv.contourArea))

        area_img = im.shape[0] * im.shape[1]
        #print(area/area_img)
        if width==0 or height==0 or area/area_img<0.02 or area/area_img>0.15:
            print("Cannot detect cigarettes or IQOS for this image.")
            class_name = 'Not Detected'
            score=0
            return score, class_name
        score = height / width


        if score < 1:
            score = 1 / score
        elif score==1:
            print("Cannot detect cigarettes or IQOS for this image.")
            class_name ='Not Detected'
            score=0
            return score, class_name
        if score < threshold:
            class_name = 'Cigarette'
        else:
            class_name = 'IQOS'

        print("The ratio is", score)
        print("The input image is classified as", class_name)
    else:
        print("Cannot detect cigarettes or IQOS for this image.")
        class_name = 'Not Detected'
        score=0
        return score, class_name

    cig_threshold_values=[4.07, 4.08, 3.12, 2.68, 3.40, 2.34, 3.4, 5.18, 1.95, 5.4]
    iqos_threshold_values=[4.93, 5.59, 6.18, 4.09, 4.64, 6.07, 4.33, 4.49, 4.33, 3.65]
    #print(np.mean(cig_threshold_values))
    #print(np.mean(iqos_threshold_values))
    #print(np.mean(cig_threshold_values)/2+np.mean(iqos_threshold_values)/2)


    return score, class_name

