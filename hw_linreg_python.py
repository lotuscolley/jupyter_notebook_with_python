#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the dataset
import pandas as pd
import sys

dataset = pd.read_csv(sys.argv[1])



# In[3]:


#Fitting Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])


# In[4]:


#Visualizing the Linear Regression results
import matplotlib.pyplot as plt
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("dataset.png")
plt.clf()


# In[5]:


#Visualizing the Linear Regression results
import matplotlib.pyplot as plt
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("linearregression.png")
plt.clf()









