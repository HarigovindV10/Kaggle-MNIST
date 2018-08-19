
# coding: utf-8

# In[44]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[45]:



# In[46]:




# In[47]:


test=pd.read_csv("test.csv")


# In[48]:


train=pd.read_csv("train.csv")


# In[49]:


train.head()


# In[50]:


train_x=train.iloc[:,0]


# In[51]:


train_x.head()


# In[52]:


train_y=train.iloc[:,1:]


# In[53]:


train_y.head()


# In[54]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_y,train_x)


# In[58]:


pred=clf.predict(test)


# In[67]:


f=open("submission.csv","w")
f.write("ImageId,Label\n")
for i in range(len(pred)):
    f.write(str(i+1)+","+str(pred[i])+"\n")
f.close()


# In[68]:


df=pd.read_csv("submission.csv")
df.head()

