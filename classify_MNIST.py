# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:45:33 2017

@author: Jing Wang
"""

from tensorflow.examples.tutorials.mnist import input_data
from sklearn import svm
from numpy import concatenate, mean
import time

# load the MNIST data by TensorFlow
mnist=input_data.read_data_sets("MNIST_data/", one_hot=False)

image_train=mnist.train.images
image_validation=mnist.validation.images
image_test=mnist.test.images

label_train=mnist.train.labels
label_validation=mnist.validation.labels
label_test=mnist.test.labels

# merge the training and validation datasets
image_train=concatenate((image_train, image_validation), axis=0)
label_train=concatenate((label_train, label_validation), axis=0)

# record time
time_start=time.time() 

# linear SVM classifier by Scikit-learn
C=1.0 # SVM regularization parameter
clf=svm.SVC(kernel='linear', C=C)
svc=clf.fit(image_train, label_train) # train
label_predict=svc.predict(image_test) # predict

# accuracy
accuracy=mean((label_predict==label_test)*1)
print('Accuracy: %0.4f.' % accuracy)

# time used
time_end=time.time()
print('Time to classify: %0.2f minuites.' % ((time_end-time_start)/60))

# # Output: 
# Accuracy: 0.9404. 
# Time to classify: 12.75 minuites. 
