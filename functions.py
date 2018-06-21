# import the necessary packages


from imutils import paths
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import confusion_matrix

import numpy as np
import imutils
import cv2
import os


def image_to_feature_vector(image, size=(32, 32)):

    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


def arguments (path):
    # construct the argument parse and parse the arguments


# grab the list of images that we'll be describing
    imagePaths = list(paths.list_images(path))
    return imagePaths
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list


def reading_images(imagePaths):
    rawImages = []
    features = []
    labels = []
# loop over the input images
    for (i, imagePath) in enumerate(imagePaths):

        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]

    # extract raw pixel intensity "features", followed by a color
    # histogram to characterize the color distribution of the pixels
    # in the image
        pixels = image_to_feature_vector(image)
        hist = extract_color_histogram(image)

    # update the raw images, features, and labels matricies,
    # respectively
        rawImages.append(pixels)
        features.append(hist)

        labels.append(label)



# show some information on the memory consumed by the raw images
# matrix and features matrix
    rawImages = np.array(rawImages)
    features = np.array(features)
    labels = np.array(labels)
    labels = column_or_1d(labels,warn=True)

    labels = label_binarize(labels.ravel(), classes=['1', '33'])  # ham will be 0 and spam will be 1

    return rawImages, features, labels


def training_testing(model ,feature, true_labels, imagePaths):
    precision_scorer = make_scorer(precision_score, pos_label=1)
    recall_scorer = make_scorer(recall_score, pos_label=1)
    accuracy = cross_val_score(model, feature, true_labels, cv=10)
    accuracy = np.mean(accuracy)
    precision = cross_val_score(model, feature, true_labels, cv=10, scoring=precision_scorer)
    precision = np.mean(precision)
    recall = cross_val_score(model, feature, true_labels, cv=10, scoring=recall_scorer)
    recall = np.mean(recall)
    print("For given model:")
    print("Accuracy = ", accuracy)
    print("Recall= ", recall)
    print("Precision= ", precision)
    predictions = cross_val_predict(model, feature, true_labels, cv=10)
    misclassified = (predictions.ravel() != true_labels.ravel())
    misclassified_ind = []
    iqos_classified=(predictions.ravel()==1)
    iqos_classified_img_list=[]
    for i in range(len(true_labels.ravel())):
        if iqos_classified[i]==True:
            iqos_classified_img_list=np.append(iqos_classified_img_list,imagePaths[i])
    print(iqos_classified_img_list)
    conf_mat = confusion_matrix(true_labels, predictions)
    print(conf_mat)
    for pred in range(len(true_labels.ravel())):
        if misclassified[pred] == True:
            misclassified_ind = np.append(misclassified_ind, int(pred))
    #misclassified_path = showing_misclassified_images(misclassified_ind, imagePaths)
    misclassified_path=[]
    return accuracy, precision, recall, misclassified_ind, misclassified_path, iqos_classified_img_list


def showing_misclassified_images(indices, imagePaths):
    misclassified_path=[]
    for i in indices:
        i = int(i)
        print("Name of the image= ", imagePaths[i])
        misclassified_path = np.append(misclassified_path, imagePaths[i])
    return misclassified_path

