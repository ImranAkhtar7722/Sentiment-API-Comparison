#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


ls = pd.DataFrame()


# In[3]:


model_results = pd.read_csv('actual_results.csv')


# In[4]:


model_data = model_results['results']


# In[5]:


print(model_data)


# In[6]:


model_data = model_data.mask(model_data > 2.5, 'positive')


# In[8]:


print(model_data)


# In[9]:


for i in model_data:
        model_data = model_data.replace(2.5,'neutral')


# In[10]:


print(model_data)


# In[30]:


model_data.iloc[0:175]


# In[39]:


for i in model_data:
        model_data = model_data.replace(2.1,'negative')
        model_data = model_data.replace(2.3,'negative')
        model_data = model_data.replace(2.36,'negative')
        model_data = model_data.replace(2.44,'negative')
        model_data = model_data.replace(2.43,'negative')
        model_data = model_data.replace(2.4,'negative')
        model_data = model_data.replace(1.83,'negative')
        model_data = model_data.replace(2.15,'negative')
        model_data = model_data.replace(2.37,'negative')
        model_data = model_data.replace(1.23,'negative')
        model_data = model_data.replace(2.41,'negative')
        model_data = model_data.replace(2,'negative')
        model_data = model_data.replace(2.35,'negative')
        model_data = model_data.replace(2.17,'negative')
        model_data = model_data.replace(1.23,'negative')
        model_data = model_data.replace(1.41,'negative')
        model_data = model_data.replace(1.23,'negative')
        model_data = model_data.replace(2.38,'negative')
        model_data = model_data.replace(2.27,'negative')
        model_data = model_data.replace(2.13,'negative')
        model_data = model_data.replace(2.23,'negative')
        model_data = model_data.replace(2.32,'negative')
        model_data = model_data.replace(2.39,'negative')
        model_data = model_data.replace(1.77,'negative')
        model_data = model_data.replace(2.25,'negative')
        model_data = model_data.replace(0.5,'negative')
        model_data = model_data.replace(1.77,'negative')


# In[40]:


print(model_data)


# In[41]:


test_results = pd.read_csv('test_results.csv')
test_data = test_results['sentiment']
print(test_data)


# In[42]:


test_data = test_data.head(175)
print(test_data)


# In[43]:


from sklearn import metrics
metrics.accuracy_score(test_data,model_data)*100


# In[ ]:





# In[ ]:




