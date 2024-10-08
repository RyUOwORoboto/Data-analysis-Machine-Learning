#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv("C:/Users/Devansh Tiwari/Desktop/video_games.csv")


# In[4]:


df


# In[5]:


df.info()


# In[5]:


df.index


# In[6]:


df.describe() #gives analysis of the data given in statistical form


# In[7]:


df.columns


# In[8]:


df.head


# In[9]:


df.dtypes


# In[10]:


df.iloc[3] #tells whole information about the specific row


# In[11]:


df.loc[4:8,['Title','Release.Year']] #taking specifications


# In[12]:


df['Release.Year']>2004 # in this it gave me info of how many games were released after 2004 as a true or false


# In[13]:


df.loc[df['Release.Year']>2004] # in this .loc gave me the whole information about the game which had ther release year greater than 2004


# In[14]:


df.loc[df['Release.Year']>2004 , ['Release.Year' , 'Features.Max Players']] #getting into more specifications 


# In[15]:


df.rename(
    columns={
        'Release.Year':'Year Of Release' ,
        'Features.Max Players':'Max no.of Players' ,
        'Title': 'Game Name'},
    
)#to change row(s) and column(s) names


# In[16]:


Year of Release.mean()


# In[17]:


df.plot()


# In[18]:


plt.plot(df.index ,df['Metrics.Sales'])


# In[19]:


# functions in pandas to handle null values (they all work for the dataframes as well)
#pd.isnull(np.nan)
#pd.isna(np.nan)
#pd.isnull(None)
#pd.isna(None)
#pd.isnull().sum()
#pd.notnull(None)


# In[20]:


df.dropna(how='all')


# In[21]:


df.fillna(method='bfill') #axis=0(vertical fillin)  if axis= 1(horizontal fillin)


# In[22]:


#pd.isnull().sum()


# In[23]:


df.replace({
    'Features.Max Players':
        {1:2,
        2:1
    },
    'Metadata.Genres':{
        'Action':'Kombat',
        'Adventure':'Explore',
        'Racing':'Thrilling'
    }
    
    
})


# In[24]:


df


# In[25]:


df.dtypes


# In[26]:


plt.scatter(df['Length.Main Story.Polled'] , df['Metrics.Review Score'] ,  c=df['Length.Main Story.Polled'], 
            s=df['Metrics.Review Score'] , cmap='Spectral' )
plt.title('Scatter Plot')
plt.xlabel('Length.Main Story.Polled')
plt.ylabel('Metrics.Review Score')


# In[27]:


plt.bar(df['Length.Main Story.Polled'] , df['Metrics.Review Score']) #,  c=df['Length.Main Story.Polled'], 
            #s=df['Metrics.Review Score'] )
plt.title('Bar Chart')
plt.xlabel('Length.Main Story.Polled')
plt.ylabel('Metrics.Review Score')


# In[8]:


plt.hist(df['Metrics.Review Score']) #df['Metrics.Review Score']) #,  c=df['Length.Main Story.Polled'], 
            #s=df['Metrics.Review Score'] )
plt.title('Histogram')
plt.xlabel('Length.Main Story.Polled')
plt.ylabel('Metrics.Review Score')


# In[ ]:




