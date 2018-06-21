
import function_3class
from function_3class import training_testing, reading_images
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
imagePaths = function_3class.arguments(path='/home/obsatir/Desktop/finalset_3class/merged')

rawImages, features, labels = reading_images(imagePaths)
labels = labels.ravel()

# kNN Model
model0 = KNeighborsClassifier(n_neighbors=5,metric='manhattan')

# SVM Model
model1 = SVC(C=50, kernel="linear")

# LR Model
model2 = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear')
#

model3 = MLPClassifier(activation='logistic', hidden_layer_sizes=40, solver='lbfgs',random_state=0)

accuracy, ind, mis_path,conf_mat= training_testing(model1, features, labels, imagePaths)

#plt.imshow(np.reshape(rawImages[int(ind[0])],(32,32,3)))
#plt.show()