############### number pattern predicting ####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets      # load dataset
from sklearn import svm

d=datasets.load_digits()
n=-10
x,y=d.data[:n],d.target[:n]
len(d.data),x,y
clf=svm.SVC(gamma=0.0001,C=100)
clf.fit(x,y)                     # fit traing data set
p=-5                             # p here going to select number in data set
print("Prdicted: ",clf.predict([d.data[p]]))
plt.imshow(d.images[p],cmap=plt.cm.gray_r,interpolation='nearest')
plt.show()
