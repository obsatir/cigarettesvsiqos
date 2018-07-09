from imutils import paths
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import column_or_1d
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
    labels = label_binarize(labels.ravel(), classes=['1', '33']) #iqos will be positive class.
    labels=labels.ravel()
    return rawImages, features, labels


def training_testing(model ,feature_train, true_labels, imagePaths, feature_test):
#Train and test the model

    model.fit(feature_train,true_labels)
    predictions = model.predict(feature_test)
#Get the images that are classified as IQOS(White) for second method.

    iqos_classified=(predictions.ravel()==1)
    iqos_classified_img_list=[]
    for i in range(feature_test.shape[0]):
        if iqos_classified[i]==True:
            iqos_classified_img_list=np.append(iqos_classified_img_list,imagePaths[i])

    return predictions, iqos_classified_img_list



