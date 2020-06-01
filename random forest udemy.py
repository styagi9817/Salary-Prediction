#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('C:\\Users\\Shubham PC\\Desktop\\Data Science Notes\\original\\Machine Learning A-Z (Codes and Datasets)\\Part 2 - Regression\\Section 9 - Random Forest Regression\\Python\\Position_Salaries.csv')
df


# In[3]:


x=df.iloc[:, 1:-1].values
y=df.iloc[: ,-1].values


# In[4]:


x


# In[5]:


y


# In[6]:


from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=10,random_state=0)
reg.fit(x,y)


# In[11]:


y_pred=reg.predict([[6.5]])
y_pred


# In[13]:


reg.score([[y,y_pred]])


# In[ ]:




