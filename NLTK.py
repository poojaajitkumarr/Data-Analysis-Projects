#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nltk')


# In[12]:


import nltk


# In[18]:


messages = [line.rstrip() for line in open(r"C:\Users\pxxjx\OneDrive\Documents\Udemy\SMSSpamCollection")]


# In[19]:


len(messages)


# In[20]:


messages[50]


# In[21]:


for mess_no,messages in enumerate(messages[:10]):
    print(mess_no,messages)
    print('\n')


# In[22]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install seaborn')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


messages = pd.read_csv(r"C:\Users\pxxjx\OneDrive\Documents\Udemy\SMSSpamCollection", sep='\t', names=['label','message'])


# In[26]:


messages.head()


# messages.describe()

# In[27]:


messages.describe()


# In[28]:


messages.groupby('label').describe()


# In[29]:


messages['length'] = messages['message'].apply(len)


# In[30]:


messages.head()


# In[31]:


messages['length'].plot.hist(bins=50)


# In[34]:


messages[messages['length']==910]['message'].iloc[0]


# In[35]:


messages.hist(column='length',by='label',bins=30)


# In[38]:


messages.hist(column='length',by='label',bins=30,figsize=(12,5))
#like subplots here and 12 -- extends figure horizontally 5-- vertically


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




