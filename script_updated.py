import functions_histogram_classification
from ratio_estimator import iqos_or_cig
from functions_histogram_classification import training_testing, reading_images
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
#imagePaths is where the training dataset is located. The images must be renamed as "{class}.{image_num}.jpg"
imagePaths = functions_histogram_classification.arguments(path='/home/obsatir/Desktop/finalsey/merged')

rawImages, features, labels = reading_images(imagePaths)

#imagePaths_test is where the images that are desired to be classified are located. It does not have to follow
#a specific renaming rule.

imagePaths_test = functions_histogram_classification.arguments(path='/home/obsatir/Desktop/1_ratio')
rawImages_test, features_test, _ = reading_images(imagePaths_test)

# kNN Model
model0 = KNeighborsClassifier(n_neighbors=5,metric='manhattan')

# SVM Model
model1 = SVC(C=50, kernel="linear")

# LR Model
model2 = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear')

# MLP
model3 = MLPClassifier(activation='logistic', hidden_layer_sizes=40, solver='lbfgs',random_state=0)

#Below, the model is trained and tested, the images that are classified as HEETS and all of the predictions are returned.
predictions,iqos_classified = training_testing(model3, features, labels, imagePaths_test,features_test)
class_names=[]


##Merged approach - Take every image that classified as IQOS(White) and further classify with ratio estimator.
dictionary= dict(zip(imagePaths_test,predictions))#This dict contains keys, which are the paths for images and its values
#are 0 for Cigarettes, 1 for HEETS and 2 for Not Classified.

for paths in iqos_classified:
    #print(paths)
    score,class_name=iqos_or_cig(paths)
    class_names=np.append(class_names,class_name)
    if class_name=='Cigarette':
        dictionary[paths]=0
    elif class_name=='IQOS':
        dictionary[paths]=1
    else:
        dictionary[paths]=2

print(dictionary)