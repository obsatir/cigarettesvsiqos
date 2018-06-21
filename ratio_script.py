import numpy as np
from width_height_ratio import iqos_or_cig
import os


path='/home/obsatir/Desktop/finalsey/33/'
iqos_or_cig(path)
images_jpg=os.listdir(path)
print(images_jpg)
class_names=[]
for images in images_jpg:
    print(images)
    path2=path+images
    score,class_name=iqos_or_cig(path2)
    class_names=np.append(class_names,class_name)
print(np.count_nonzero(class_names=='Cigarette'))
print(np.count_nonzero(class_names=='IQOS'))
print(np.count_nonzero(class_names=='Not Detected'))