
import functions
from width_height_ratio import iqos_or_cig
from functions import training_testing, reading_images
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
imagePaths = functions.arguments(path='/home/obsatir/Desktop/finalsey/merged')

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

accuracy, precision, recall, ind, mis_path,iqos_classified = training_testing(model3, features, labels, imagePaths)
#print(accuracy)
class_names=[]
##Merged approach
for paths in iqos_classified:
    print(paths)
    score,class_name=iqos_or_cig(paths)

    class_names=np.append(class_names,class_name)

print(np.count_nonzero(class_names=='Cigarette'))
print(np.count_nonzero(class_names=='IQOS'))
print(np.count_nonzero(class_names=='Not Detected'))