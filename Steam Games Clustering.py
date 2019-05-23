#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

# setting path

path = 'C:\Python\webscrape'
os.chdir(path) 


# In[2]:


# reading from csv

data = pd.read_csv('steam_sales May, 12, 2019.txt', delimiter = '*', error_bad_lines = False)


# In[3]:


data.head()


# In[4]:


# converting discount to decimal and replacing NA's with 0

data['discount'] = data['discount'].str.rstrip('%').astype('float') / 100.0 *(-1)
data[['discount','discounted price']] = data[['discount','discounted price']].fillna(value = 0)


# In[5]:


data.head()


# In[6]:


# converting 'free to play' to 0 and stripping unecessary strings from original price

data = data.replace(to_replace = ['Free to Play', 'Free To Play', 'Free', 'Play for Free!', 
                                  'Free Demo', 'Free Movie', '1 Season', 'Third-party',
                                  'Play Now', 'Free Mod', 'From CDN$ 25.60', 'Install',
                                  'CDN$ 1,200.76', 'From CDN$ 25.31', 'From CDN$ 19.20'], value = np.nan)

data['original price'] = data['original price'].str.lstrip('CDN$ ').astype('float')
data['original price'] = data['original price'].fillna(value = 0)


# In[7]:


data.head()


# In[8]:


# formatting discounted price

data = data.replace(to_replace = 'CDN$ 1,063.85							', value = np.nan)
data['discounted price'] = data['discounted price'].str.lstrip('CDN$ ').str.rstrip('\t\t\t\t\t\t\t').astype('float')
data['discounted price'] = data['discounted price'].fillna(value = data['original price'])


# In[9]:


data.head()


# In[10]:


# filling in 'None' reviews and one hot encoding to handle categorical review variable

# to_cut = data['reviews'].unique().tolist()
# to_cut_df = pd.DataFrame(data = {'terms': to_cut})
# to_cut_df.to_csv('to_cut')

# review_order = {'Negative': 0, 'Mostly Negative': 1, 'Mixed': 2,
#               'Mostly Positive': 3, 'Positive': 4, 'Very Positive': 5, 
#               'Overwhelmingly Positive': 6}

# data['reviews label'] = data['reviews'].map(review_order)

data['reviews'] = data['reviews'].fillna(value = 'None')
data = pd.concat([data ,pd.get_dummies(data['reviews'], drop_first = True)],axis=1)
data.drop(['reviews'], axis=1, inplace=True)


# In[11]:


data.head()


# In[12]:


# exploratory analysis

data.describe()


# In[13]:


# training and testing set

train, test = train_test_split(data, test_size=0.2)


# In[14]:


train.describe()


# In[15]:


test.describe()


# In[16]:


# visualization

plt.scatter(train['original price'], train['None'])


# In[17]:


# clustering


# In[ ]:




