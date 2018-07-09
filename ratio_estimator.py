import cv2 as cv
import numpy as np

def iqos_or_cig (path):
    im = cv.imread(path, 0)
    ret, iqos2 = cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    count_black = np.count_nonzero(iqos2 == 0)
    count_white = np.count_nonzero(iqos2 == 255)
    if count_white > count_black: #Checking whether the segmented part is white or black (It must be white).
        for i in range(iqos2.shape[0]):
            for j in range(iqos2.shape[1]):
                if iqos2[i][j] == 255:
                    iqos2[i][j] = 0
                elif iqos2[i][j] == 0:
                    iqos2[i][j] = 255

    _, cnt, hierarchy = cv.findContours(iqos2, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)  # Getting contours

    if len(cnt) != 0:
        rect = cv.minAreaRect(max(cnt, key = cv.contourArea)) #Getting the width and height of the bounding box of
                                                              #contour with max area, which is the object.
        width=rect[1][0]
        height=rect[1][1]
        threshold = 3.5
        area = cv.contourArea(max(cnt, key=cv.contourArea))
        area_img = im.shape[0] * im.shape[1]

        if width==0 or height==0 or area/area_img<0.02 or area/area_img>0.15: #Checking whether the contour is well
                                                                              #detected or not.
            class_name = 'Not Detected'
            score=0
            return score, class_name
        score = height / width


        if score < 1:

            score = 1 / score

        elif score==1:
            class_name ='Not Detected'
            score=0
            return score, class_name

        if score < threshold:
            class_name = 'Cigarette'
        else:
            class_name = 'IQOS'

    else:
        class_name = 'Not Detected'
        score=0
        return score, class_name



    return score, class_name

