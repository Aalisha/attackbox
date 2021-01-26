import numpy as np
import os
from PIL import Image
import sklearn

#Results
# Model - detectron2
#num_images:  mean L2 error:   median L2 error:  max L2 error: 
#99 0.04922374562990482 0.04647780864865012 0.13123548601766416

#Model - fasterrcnn
#num_images:  mean L2 error:   median L2 error:  max L2 error: 
#123 0.014985392978783613 0.011414213984204792 0.05417457339791851

model = 'fasterrcnn'
images = os.listdir(images_path)
adv_images = os.listdir(adv_images_path)
l2_error = []

for img_name, adv_img_name in zip(images, adv_images):
    img = Image.open(os.path.join(images_path, img_name)).convert('RGB')
    adv_img = Image.open(os.path.join(adv_images_path, adv_img_name)).convert('RGB')
    
    arr_img = np.array(img)
    arr_adv_img = np.array(adv_img)

    norm_img = (arr_img - arr_img.min())/(arr_img.max() - arr_img.min())
    norm_adv_img = (arr_adv_img - arr_adv_img.min())/(arr_adv_img.max() - arr_adv_img.min())

    l2_error.append(np.mean((np.square(norm_img - norm_adv_img))))
    
    #normalize_img = sklearn.preprocessing.normalize(np.array(img))
    #normalize_adv_img = sklearn.preprocessing.normalize(np.array(adv_img))
print('Model - fasterrcnn')
print('num_images: ', 'mean L2 error:  ', 'median L2 error: ', 'max L2 error: ')
print(len(images), np.mean(l2_error), np.median(l2_error), np.max(l2_error))