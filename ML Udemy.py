#!/usr/bin/env python
# coding: utf-8

# # MACHINE LEARNING ALGORITHMS

# # Linear Regression

# In[1]:


get_ipython().system('pip install pandas')


# In[ ]:





# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv(r"C:\Users\pxxjx\OneDrive\Documents\Udemy\USA_Housing.csv")


# In[4]:


get_ipython().system('pip install numpy')


# In[5]:


import numpy as np


# In[6]:


get_ipython().system('pip install seaborn')
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df.head()


# In[9]:


df.info()
df.describe()


# In[10]:


df.columns


# In[11]:


sns.pairplot(df)


# In[13]:


sns.displot(df['Price'])


# In[14]:


sns.heatmap(df.corr(),annot=True)


# In[15]:


X = df[['lotsize', 'bedrooms', 'bathrms', 'stories']]
y = df['Price']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()
#Creating linear regression object


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


print(lm.intercept_)


# In[ ]:


lm.coef_


# In[ ]:


X_train.columns


# In[ ]:


cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
cdf
#increase in one unit of lotsize implies 5.4 times 

Predictions
# In[ ]:


predictions = lm.predict(X_test)
#in predict, the model needs data that it has never seen before. 
#Hence, the test. 


# In[ ]:


predictions


# In[ ]:


y_test


# In[ ]:


#to compare if our actual data and predicted data is close enough, we use plots 
plt.scatter(y_test, predictions)


# In[ ]:


sns.displot((y_test-predictions),kde=True)


# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.mean_absolute_error(y_test,predictions)


# In[ ]:


np.sqrt(metrics.mean_absolute_error(y_test,predictions))


# # Logistic reggression

# In[1]:


get_ipython().system('pip install pandas')
import pandas as pd
tit = pd.read_csv(r"C:\Users\pxxjx\OneDrive\Documents\Udemy\titanic_train.csv")


# In[2]:


tit


# In[3]:


get_ipython().system('pip install seaborn')
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 


# In[4]:


tit.head()


# In[5]:


print(tit.columns)


# In[6]:


sns.heatmap(tit.isnull())


# In[7]:


sns.heatmap(tit.isnull(),cbar=False,yticklabels=False,cmap='viridis')


# In[8]:


CountPlot = sns.countplot(x='Survived',hue='Sex',data=tit)


# In[9]:


CountPlot = sns.countplot(x='Survived',hue='Pclass',data=tit)


# In[10]:


tit['Age'].plot.hist(bins=30)


# In[11]:


tit.info()


# In[12]:


sns.countplot(x='SibSp',data=tit)


# In[13]:


tit['Fare'].plot.hist(bins=40,figsize=(10,4))


# In[14]:


get_ipython().system('pip install cufflinks')
import cufflinks as cf
cf.go_offline


# In[15]:


#tit['Fare'].iplot(kind='hist',bins=40) - can't get it right


# In[16]:


plt.figure(figsize = (10,7))
sns.boxplot(x='Pclass',y='Age',data=tit)


# In[17]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else: 
            return 24
    else:
        return Age


# In[18]:


tit['Age']=tit[['Age','Pclass']].apply(impute_age,axis=1)


# In[19]:


sns.heatmap(tit.isnull(),cbar=False,yticklabels=False,cmap='viridis')


# In[20]:


tit.drop('Cabin',axis=1,inplace=True)


# In[21]:


#Create categorical variables into dummy variables or ML can't take it as inputs -- zero or one values 


# In[22]:


sex = pd.get_dummies(tit['Sex'],drop_first=True)


# In[23]:


embark = pd.get_dummies(tit['Embarked'],drop_first=True)


# In[24]:


tit=pd.concat([tit,embark,sex],axis=1)


# In[25]:


tit.head()


# In[26]:


tit.drop(['Name','Sex','Embarked','Ticket'],axis=1,inplace=True)


# # Decision Trees and Random Forest

# In[27]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install seaborn')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


dec=pd.read_csv(r"C:\Users\pxxjx\OneDrive\Documents\Udemy\Decision tree")


# In[29]:


dec.info()
dec.describe()
dec.columns
dec.head()


# In[30]:


sns.pairplot(dec,hue='Kyphosis')


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X=dec.drop('Kyphosis',axis=1)
y=dec['Kyphosis']


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[34]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[35]:


predictions = dtree.predict(X_test)


# In[36]:


from sklearn.metrics import classification_report,confusion_matrix


# In[37]:


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[38]:


from sklearn.ensemble import RandomForestClassifier


# In[39]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
predict=rfc.predict(X_test)
print(confusion_matrix(y_test,predict))
print(classification_report(y_test,predict))


# # Support Vector Machine

# In[40]:


from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
cancer.keys()
#identify if the data will be tumour or benign using SVM


# In[41]:


df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[42]:


df_feat.head(2)


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


X=df_feat
y=cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[45]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)


# In[46]:


predctions_cancer = model.predict(X_test)


# In[47]:


from sklearn.metrics import classification_report, confusion_matrix


# In[48]:


print(confusion_matrix(y_test,predctions_cancer))
print('\n')
print(classification_report(y_test,predctions_cancer))


# In[52]:


get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install --upgrade scikit-learn')

from sklearn.model_selection import GridSearchCV


# In[ ]:


#Can't get the module 
param_grid = {'C:'}


# In[53]:


from sklearn.datasets import make_blobs


# In[54]:


data = make_blobs(n_samples=200,n_features=2,centers=4,random_state=101)


# In[55]:


plt.scatter(data[0][:,0],data[0][:,1],c=data[1])


# In[56]:


from sklearn.cluster import KMeans


# In[57]:


kmeans = KMeans(n_clusters=3)


# In[58]:


kmeans.fit(data[0])


# In[59]:


fig , (ax1,ax2) = plt.subplots(1,2, sharey=True, figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_)

ax2.set_title('Original')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1])


# # Pca

# In[60]:


from sklearn.datasets import load_breast_cancer


# In[61]:


cancer = load_breast_cancer()


# In[63]:


type(cancer)
#works like a dictionary 


# In[64]:


#Calling keys of the dictionary 
cancer.keys()


# In[65]:


print(cancer['DESCR'])


# In[66]:


#find most important component in the dataset that causes variance. Not classification


# In[67]:


df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[68]:


from sklearn.preprocessing import StandardScaler


# In[69]:


scaler = StandardScaler()


# In[71]:


scaler.fit(df)


# In[72]:


scaled_data=scaler.transform(df)


# In[73]:


#PCA
from sklearn.decomposition import PCA


# In[74]:


pca = PCA(n_components=2)
#Keeping two ccomponents


# In[75]:


pca.fit(scaled_data)


# In[76]:


x_pca = pca.transform(scaled_data)


# In[77]:


x_pca.shape


# In[83]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('First Component')
plt.ylabel('Second Component')
#target implies benign or malignant


# # NLP

# In[1]:


get_ipython().system('pip install nltk')


# In[2]:


#Spam detection filter is to be created 


# In[3]:


import nltk 


# In[ ]:


#nltk.download_shell()


# In[ ]:




