#!/usr/bin/env python
# coding: utf-8

# In[1]:


#check the dataset present in the file
import os
print(os.listdir())


# In[2]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


#import dataset
data = pd.read_csv('housingData-Real.csv')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


#select your columns
livingspace = data['sqft_living']
price = data['price']


# In[7]:


#convert livingspace into 20 matrix
X= np.array(livingspace).reshape(-1,1)


# In[8]:


y = np.array(price)


# In[9]:


X


# In[10]:


#convert the data into test and training
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33)


# In[11]:


X_test


# In[12]:


#pass the data in the linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[13]:


y_predictor = regressor.predict(X_test)


# In[14]:


y_predictor


# In[15]:


#this Prediction id more helpful with Graphs/plots


# In[16]:


#plot for training dataset
plt.scatter(X_train,y_train)
plt.plot(X_train,regressor.predict(X_train))
plt.title('Training graph for Housing')


# In[17]:


#plot the test dataset
plt.scatter(X_test,y_test)
plt.plot(X_train,regressor.predict(X_train),color = 'red')
plt.title('test graph for Housing')
plt.xlabel('Living space')
plt.ylabel('Pricing')


# In[18]:


year = data['bedrooms']
price = data['price']


# In[19]:


type(year)


# In[20]:


X = np.array(year).reshape(-1,1)


# In[21]:


y =np.array(price)


# In[22]:


X


# In[23]:


y


# In[24]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[25]:


X_train


# In[26]:


y_train


# In[27]:


from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_train,y_train)


# In[28]:


predictor = regr.predict(X_test)


# In[29]:


predictor


# In[30]:


plt.scatter(X_train,y_train)
plt.plot(X_train,regr.predict(X_train),color = 'red')
plt.title('Training graph of housing')
plt.xlabel('Bedrooms')
plt.ylabel('Price')


# In[31]:


plt.scatter(X_test,y_test)
plt.plot(X_train,regr.predict(X_train),color = 'red')
plt.title('Test graph of housing')
plt.xlabel('Built in Year')
plt.ylabel('Price')


# ## Multiple Linear Regression
# 

# In[34]:


data.info()


# In[35]:


data.head()


# In[42]:


data['date']=pd.to_numeric(data['date'])


# In[43]:


data.info()


# In[74]:


y=data.iloc[:,[2]].values
X=data.iloc[:,[x for x in range(21) if x!=2]].values


# In[75]:


X.info()


# In[76]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[77]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[78]:


y_predictor = regressor.predict(X_test)


# In[79]:


y_predictor


# In[72]:


print(y_test,y_predictor
     )


# In[80]:


np.set_printoptions(precision =2)
print(np.concatenate((y_predictor.reshape(len(y_predictor),1),y_test.reshape(len(y_test),1)), 1))


# In[82]:


regressor.coef_


# In[83]:


regressor.intercept_


# In[84]:


plt.plot(X_test,y_test,color = 'red')
plt.plot(X_test,y_predictor,color = 'green')


# In[ ]:




