#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')


# In[2]:


get_ipython().system('pip install numpy')


# In[3]:


import numpy as np


# In[4]:


import pandas as pd


# In[5]:


df = pd.read_excel(r'C:\Users\pxxjx\Downloads\SMS ex 1.xlsx', sheet_name='Sheet2')


# In[6]:


print(df)


# In[7]:


df.reset_index(drop='true')


# In[8]:


sf=df.dropna()


# In[9]:


sf


# In[10]:


df['RN']


# # MATPLOTLIB
# 

# In[11]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


df.columns.str.strip()


# In[13]:


df.head()


# In[14]:


df.columns.tolist()


# In[15]:


fig = plt.figure()
axes=fig.add_axes([1,1,1,1])
axes.plot(sf['Week '],sf['Revenue'],color='magenta', linewidth=1,alpha=0.5,ls='-',marker='*',markerfacecolor='red',
          markeredgewidth='2',markeredgecolor='pink')
axes.set_xlabel('WEEK')
axes.set_ylabel('REVENUE')
axes.set_title('SMS EX 1')
#alpha is for transparency #can go with lw for linewidth 


# In[16]:


sf['Revenue'].max()


# In[17]:


sf.loc[2]


# In[18]:


import seaborn as sns


# In[19]:


get_ipython().system('pip install seaborn')


# In[20]:


import seaborn as sns


# In[24]:


x=sf['Week ']
y=sf["Revenue"]
sns.jointplot(x=x,y=y,kind='hex')


# In[27]:


import pandas as pd
df = pd.read_excel('https://query.data.world/s/snavhuasii3naqxhuhuvsqce5j4xhv?dws=00000')


# In[28]:


df


# In[33]:


sns.barplot(x=df['Country'],y=df['Cost'],estimator=np.std)
plt.tight_layout()
plt.show()


# In[38]:


sns.boxplot(x=df['Cost'],y=df['Country'],hue=df['Category'])


# In[40]:


get_ipython().system('pip install cufflinks')
import cufflinks as cf
from plotly import __version__
cf.go_offline()
df.iplot()


# In[1]:


get_ipython().system('pip install scikit-learn')


# In[ ]:




