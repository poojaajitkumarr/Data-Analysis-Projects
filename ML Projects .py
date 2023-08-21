#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install seaborn')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv(r"C:\Users\pxxjx\OneDrive\Documents\Udemy\Fish-LinearReg")


# In[3]:


df.columns


# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


df['Species'].info()


# In[7]:


def impute_species(cols):
    species = cols[0]
    if species == 'Bream':
        return 1
    if species == 'Roach':
        return 2
    if species == 'Whitefish':
        return 3
    if species == 'Parkki':
        return 4
    if species == 'Perch':
        return 5
    if species == 'Pike':
        return 6
    if species == 'Smelt':
        return 7
    if species == 'Catfish':
        return 8
    else:
        return np.nan


# In[8]:


df['Species'] = df[['Species']].apply(impute_species,axis=1)


# In[9]:


sns.pairplot(df)


# In[10]:


sns.displot(df['Species'],color='orange')


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X = df[['Weight', 'Length1', 'Length2', 'Length3', 'Height','Width']]
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[13]:


from sklearn.linear_model import LinearRegression 


# In[14]:


model = LinearRegression()


# In[15]:


model.fit(X_train,y_train)


# In[17]:


predict = model.predict(X_test)                          


# In[37]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[38]:


print(y_test)


# In[39]:


print(predict)


# In[43]:


plt.hist(y_test, bins=30, label='ACTUAL DATA')
plt.hist(predict, bins=30, label='PREDICTED DATA')
plt.legend(loc='upper right')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




