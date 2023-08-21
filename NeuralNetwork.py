#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install seaborn')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[2]:


df = pd.read_csv(r"C:\Users\pxxjx\OneDrive\Documents\Udemy\fake_reg.csv")


# In[3]:


df.head()


# In[4]:


sns.pairplot(df)


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


#Here .values() is used for converting X and y to numpy arrays since tensor flow works with numpy arrays
X = df[['feature1','feature2']].values
y = df['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[7]:


from sklearn.preprocessing import MinMaxScaler
#Arrange data as per min, max -- whatever is being passed thru the final network is what is scaled. Here features


# In[8]:


scaler = MinMaxScaler()


# In[9]:


scaler.fit(X_train)
#here, fit finds the SD, min and max of features


# In[10]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[11]:


get_ipython().system('pip install tensorflow')


# In[12]:


from tensorflow.keras.models import Sequential


# In[13]:


from tensorflow.keras.layers import Dense


# In[14]:


#model = Sequential([Dense(4,activation='relu'),Dense(2,activation='relu'),Dense(2)])
"""
Both are one and the same 
Above one, we've created list
"""
model = Sequential()
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1)) #--- Output layer
model.compile(optimizer='rmsprop',loss='mse')


# In[15]:


model.fit(x=X_train,y=y_train,epochs=250)


# In[16]:


loss_df = pd.DataFrame(model.history.history)
loss_df.plot()


# In[17]:


model.evaluate(X_train,y_train,verbose=0)
#returns the mse
#we can do the same for training set also


# In[18]:


test_predictions = model.predict(X_test)


# In[21]:


"""
test_predictions = test_predictions.values.reshape(330,)
pred_df = pd.DataFrame(y_test,columns=['Test True Y'])
pred_df = pd.concat([pred_df,test_predictions],axis=1)
"""


# # Part 2

# In[22]:


df = pd.read_csv(r"C:\Users\pxxjx\OneDrive\Documents\Udemy\TensorFlow_FILES\DATA\kc_house_data.csv")


# In[23]:


df.head()


# In[24]:


df.columns


# In[25]:


sns.lmplot(x='bedrooms',y='price',data=df,hue='floors')


# In[26]:


df['floors'].info()


# In[27]:


plt.figure(figsize=(15,6))
sns.displot(df['price'],kde=True)


# In[28]:


df.corr()['price'].sort_values()


# In[29]:


plt.figure(figsize=(12,6))
sns.scatterplot(x='price',y='sqft_living',data=df)


# In[30]:


plt.figure(figsize=(15,6))
sns.scatterplot(x='long',y='lat',data=df,hue='price')


# In[31]:


TopPercent = df.sort_values('price',ascending=False).iloc[216:]


# In[32]:


plt.figure(figsize=(15,6))
sns.scatterplot(x='long',y='lat',data=TopPercent,hue='price',palette='rainbow')


# In[33]:


df = df.drop('id',axis=1)


# In[34]:


df['date'] = pd.to_datetime(df['date'])


# In[35]:


df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)


# In[36]:


df.head()


# In[37]:


sns.boxplot(x='month',y='price',data=df)


# In[38]:


df = df.drop('date',axis=1)


# In[39]:


df = df.drop('zipcode',axis=1)


# In[49]:


X = df.drop('price',axis=1).values
y = df['price'].values


# In[41]:


from sklearn.model_selection import train_test_split


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[43]:


from sklearn.preprocessing import MinMaxScaler


# In[51]:


scaler = MinMaxScaler()


# In[52]:


#Transform and fit on the same step
X_train = scaler.fit_transform(X_train)


# In[53]:


#We only transform the test data so that the model does not get knowloedge about the outputs
X_test = scaler.transform(X_test)


# In[47]:


get_ipython().system('pip install tensorflow')


# In[48]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[55]:


X_train.shape


# In[57]:


model = Sequential()
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# In[58]:


model.fit(x=X_test,y=y_test,validation_data=(X_test,y_test),batch_size=128,epochs=500)


# In[60]:


losses=pd.DataFrame(model.history.history)
losses.plot()


# data is not overfitting, since the loss is not spiking after a particular point. If it spikes then, we can do "early stopping"

# In[61]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[63]:


predictions = model.predict(X_test)


# In[66]:


np.sqrt(mean_squared_error(predictions,y_test))


# In[67]:


mean_absolute_error(predictions,y_test)


# In[70]:


plt.figure(figsize=(10,5))
plt.scatter(y_test,predictions)
plt.plot(y_test,y_test,'r')


# In[71]:


single_house = df.drop('price',axis=1).iloc[0]


# In[72]:


single_house = scaler.transform(single_house.values.reshape(-1,19))


# In[73]:


model.predict(single_house)


# In[74]:


df.head(1)


# The difference between predicted value and actual value is bcs of the outliers and will have to remove the top 1 or 2 precent outliers 

# # Early stopping demonstration

# In[76]:


df = pd.read_csv(r"C:\Users\pxxjx\OneDrive\Documents\Udemy\TensorFlow_FILES\DATA\cancer_classification.csv")


# In[77]:


df.info()


# In[79]:


sns.countplot(x= df['benign_0__mal_1'])


# In[82]:


df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')


# In[85]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr())


# In[89]:


X = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values


# In[90]:


from sklearn.model_selection import train_test_split


# In[91]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[93]:


from sklearn.preprocessing import MinMaxScaler


# In[94]:


scale = MinMaxScaler()


# In[96]:


X_train = scale.fit_transform(X_train)


# In[97]:


X_test = scale.transform(X_test)


# In[98]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[99]:


X_train.shape


# In[100]:


model = Sequential()
model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(1,activation='sigmoid'))#since binary classification
model.compile(optimizer='adam',loss='binary_crossentropy')


# In[101]:


model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=600)
#epoch given 600 to illustrate over fitting


# In[102]:


losses = pd.DataFrame(model.history.history)


# In[103]:


plt.plot(losses)


# The hike of the orange line(validation loss) indicates the over fitting. Hence the training has to be stopped early.

# In[105]:


from tensorflow.keras.callbacks import EarlyStopping


# In[106]:


help(EarlyStopping)


# In[108]:


#early stopping has two steps
early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)
#check what patience is for 


# In[109]:


model = Sequential()
model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(1,activation='sigmoid'))#since binary classification
model.compile(optimizer='adam',loss='binary_crossentropy')


# In[111]:


model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=600,callbacks=[early_stop])


# In[113]:


mod_loss = pd.DataFrame(model.history.history)
mod_loss.plot()


# In[119]:


from tensorflow.keras.layers import Dropout


# In[120]:


model = Sequential()
model.add(Dense(30,activation='relu'))
# 0 means 0 layers are dropped out. 1 means all of them are 
model.add(Dropout(0.5))
model.add(Dense(15,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))#since binary classification
model.compile(optimizer='adam',loss='binary_crossentropy')


# In[121]:


model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=600,callbacks=[early_stop])


# In[123]:


model_loss = pd.DataFrame(model.history.history)
plt.plot(model_loss)


# In[125]:


predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix


# In[126]:


print(classification_report(y_test,predictions))


# In[ ]:




