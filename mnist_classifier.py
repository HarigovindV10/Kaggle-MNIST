import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import tree

test=pd.read_csv("test.csv")
train=pd.read_csv("train.csv")
train_x=train.iloc[:,0]
train_y=train.iloc[:,1:]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_y,train_x)
pred=clf.predict(test)

f=open("submission.csv","w")
f.write("ImageId,Label\n")
for i in range(len(pred)):
    f.write(str(i+1)+","+str(pred[i])+"\n")
f.close()


