#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import os
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[40]:


raw_df = pd.read_csv("C:/Users/Devansh Tiwari/Desktop/weatherAUS.csv")


# In[41]:


raw_df


# In[42]:


raw_df.info()


# In[43]:


raw_df.dropna(subset = ['RainTomorrow'] , inplace = True)


# In[44]:


plt.title('No. of Rows per year')

sns.countplot(x= pd.to_datetime(raw_df.Date).dt.year);


# In[45]:


px.scatter(raw_df, 
           title='Pressure_at_9am VS Pressure_at_3pm' ,
          x='Pressure9am' ,
          y='Pressure3pm',
          color='RainToday')


# In[46]:


year = pd.to_datetime(raw_df.Date).dt.year


# In[47]:


train_df = raw_df[year<2015]
val_df = raw_df[year==2015]
test_df = raw_df[year>2015]


# In[48]:


print('train_df.shapre: ', train_df.shape)
print('val_df.shapre: ', val_df.shape)
print('test_df.shapre: ', test_df.shape)


# In[49]:


input_cols = list(train_df.columns)[1:-1]
target_cols = 'RainTomorrow'


# In[50]:


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_cols].copy()


# In[51]:


val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_cols].copy()


# In[52]:


test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_cols].copy()


# In[53]:


numeric_cols = train_inputs.select_dtypes(include = np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


# In[54]:


print(numeric_cols)


# In[55]:


print(categorical_cols)


# In[56]:


from sklearn.impute import SimpleImputer


# In[57]:


imputer = SimpleImputer(strategy = 'mean').fit(raw_df[numeric_cols])


# In[58]:


train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])


# In[59]:


test_inputs[numeric_cols].isna().sum()


# In[60]:


from sklearn.preprocessing import MinMaxScaler


# In[61]:


scaler = MinMaxScaler().fit(raw_df[numeric_cols])


# In[62]:


train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


# In[63]:


from sklearn.preprocessing import OneHotEncoder


# In[64]:


encoder = OneHotEncoder(sparse = False , handle_unknown ='ignore').fit(raw_df[categorical_cols])


# In[65]:


encoded_cols = list(encoder.get_feature_names_out(categorical_cols))


# In[66]:


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_Inputs[categorical_cols])


# In[75]:


X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


# In[ ]:


train_targets


# In[ ]:


X_test


# In[76]:


X_val


# In[77]:


val_inputs


# In[78]:


from sklearn.tree import DecisionTreeClassifier


# In[79]:


model = DecisionTreeClassifier(random_state = 42)


# In[80]:


model.fit(X_train , train_targets)


# In[ ]:





# In[ ]:





# In[82]:


from sklearn.metrics import accuracy_score , confusion_matrix


# In[85]:


train_preds = model.predict(X_train)


# In[86]:


pd.value_counts(train_preds)


# In[88]:


train_preds


# In[87]:


train_targets


# In[89]:


accuracy_score(train_preds , train_targets)


# In[ ]:




