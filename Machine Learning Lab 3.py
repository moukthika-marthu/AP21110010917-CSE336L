#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


df = pd.read_csv('tips.csv')


# Read the tips dataset from the provided source only. Handle the missing values with the appropriate techniques.

# In[19]:


df.head()


# In[20]:


df.isnull().sum()


# In[22]:


df.isnull().sum().sum()


# In[23]:


df.info()


# In[25]:


df['total_bill'].isnull().sum()


# In[26]:


df.notnull().sum()


# In[27]:


df.notnull().sum().sum()


# In[28]:


df['total_bill'].mean()


# In[29]:


df.isnull().mean()


# In[30]:


df.isna().sum()


# In[31]:


df['total_bill']=df['total_bill'].fillna(df['total_bill'].mean())


# In[32]:


df.head(5)


# In[33]:


df.isnull().sum()


# Handle the categorical data in the tips dataset with the relevant approaches such as label-encoding, one hot encoding, ordinal encoding.

# In[35]:


df.info()


# In[36]:


from sklearn.preprocessing import OneHotEncoder
o=OneHotEncoder()


# In[37]:


o.fit_transform(df[['sex','smoker','day','time']]).toarray()


# Perform feature scaling techniques such as min-max normalization, standardization, z-score, on the tips dataset

# In[38]:


from sklearn.preprocessing import MinMaxScaler
m=MinMaxScaler()


# In[39]:


df_m=m.fit_transform(df[['total_bill','tip']])


# In[40]:


d=pd.DataFrame(df_m,columns=['total_bill','tip'])
d.sample(5)


# In[43]:


plt.hist(d['total_bill'],bins=25,color='pink',edgecolor='black')


# In[44]:


from sklearn.preprocessing import StandardScaler
s=StandardScaler()


# In[45]:


df_s=s.fit_transform(df[['total_bill','tip']])
df_s


# In[47]:


d.head()


# In[48]:


z=(d['total_bill']-d['total_bill'].mean())/d['total_bill'].std()
z


# Create a new feature representing the average tip percentage for each dining party size.

# In[49]:


df.head()


# In[50]:


df.tail()


# In[51]:


df['avg_tip']=df.groupby('size')['tip'].transform('mean')
df.head(5)


# Create a new feature based on total bill and tips if total bill is greater than 10$ and tips is greater than 3$ mark as Highest-bills-with-tips otherwise Normal-bills

# In[52]:


def categorize(total_bill,tip):
    if total_bill>10 and tip>3:
        return 'Highest-bills-with-tips'
    else:
        return 'Normal-bills'
df['bill_category']=df.apply(lambda row:categorize(row['total_bill'],row['tip']),axis=1)


# In[54]:


df.head(6)

