#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("C:/Users/Devansh Tiwari/Desktop/insurance.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.index


# In[7]:


df.columns


# In[8]:


df.head


# In[9]:


df.dtypes


# In[10]:


df.iloc[4]


# In[11]:


plotly.import express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


##############  Default Settings for the Charts we will create   ######################

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['figure.figsize'] = (11,6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[13]:


df.age.describe()


# In[14]:


fig=px.histogram(df,
                x='age',
                marginal='box',
                #nbins=50,
                title='Distribution of Age')
fig.update_layout(bargap=0.1)
fig.show()


# In[15]:


fig=px.histogram(df,
                x='bmi',
                marginal='box',
                color_discrete_sequence=['purple'],
                title='Distribution of BMI')
fig.update_layout(bargap=0.1)
fig.show()


# In[16]:


fig=px.histogram(df,
                x='charges',
                marginal='box',
                color = 'smoker',
                color_discrete_sequence=['purple' , 'red'],
                title='Annual medical charges')
fig.update_layout(bargap=0.1)
fig.show()


# In[17]:


fig=px.histogram(df,
                x='charges',
                marginal='box',
                color = 'region',
                color_discrete_sequence=['purple' , 'red' ,'blue', 'black'],
                title='Annual medical charges')
fig.update_layout(bargap=0.1)
fig.show()


# In[18]:


fig=px.histogram(df,
                x='region',
                marginal='box',
                color = 'smoker',
                color_discrete_sequence=['purple' , 'red'],
                title='regional smokers')
fig.update_layout(bargap=0.1)
fig.show()


# In[19]:


fig=px.histogram(df,
                x='charges',
                marginal='box',
                color = 'sex',
                color_discrete_sequence=['purple' , 'red'],
                title='Annual medical charges')
fig.update_layout(bargap=0.1)
fig.show()


# In[20]:


df.smoker.value_counts()


# In[21]:


px.histogram(df , x='smoker', color='sex', title='Smoker')


# In[22]:


px.histogram(df , x='sex', color='sex', title='Gender')


# In[23]:


px.histogram(df , x='region', color='sex', title='Region')


# In[24]:


fig = px.scatter(df,
                 x='age',
                 y='charges',
                 color='smoker',
                 opacity=0.8,
                 hover_data=['sex'],
                 title='Age vs Charges')


fig.update_traces(marker_size=5)
fig.show()


# In[25]:


fig = px.scatter(df,
                 x='bmi',
                 y='charges',
                 color='smoker',
                 opacity=0.8,
                 hover_data=['sex'],
                 title='BMI vs Charges')


fig.update_traces(marker_size=4)
fig.show()


# In[26]:


fig=px.histogram(df,
                x='bmi',
                y='charges',
                marginal='box',
                color = 'smoker',
                color_discrete_sequence=['purple' , 'red'],
                title='Annual medical charges')
fig.update_layout(bargap=0.1)
fig.show()


# In[27]:


df.charges.corr(df.age)


# In[28]:


df.charges.corr(df.bmi)


# In[29]:


#TO COMPARE THE CORRELATION FOR CATEGORICAL COLUMNS U NEED TO CONVERT THEM INTO NUMERIC VALUES FIRST

smoker_values = {'no':0 , 'yes':1}
smoker_numeric = df.smoker.map(smoker_values)
df.charges.corr(smoker_numeric)


# In[30]:


med_df = df.select_dtypes(include=['float64', 'int64'])
med_df.corr()


# In[31]:


#df = df.select_dtypes(include=['float64', 'int64'])

sns.heatmap(med_df.corr(), cmap='Reds', annot=True)
plt.title('Correlation Matrix')


# In[32]:


non_smoker_df = df[df['smoker'] == 'no']


# In[33]:


df.dtypes


# In[34]:


plt.title('Age VS Charges')
sns.scatterplot(data = non_smoker_df , x='age' , y='charges' , alpha=0.8, s=100)


# In[35]:


def estimate_charges(age, w , b):
    return w*age + b


# In[36]:


w=50
b=100


# In[37]:


estimate_charges(59,w,b)


# In[38]:


ages = non_smoker_df.age
ages


# In[39]:


estimated_charges = estimate_charges(ages, w ,b)
estimated_charges


# In[40]:


non_smoker_df.charges


# In[41]:


plt.plot(ages,estimated_charges,'r')
plt.xlabel('Age')
plt.ylabel('Estimated_Charges')


# In[42]:


plt.plot(ages,estimated_charges,'r')
sns.scatterplot(data = non_smoker_df , x='age' , y='charges' , alpha=0.8, s=100)
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Age VS Charges')
plt.legend(['Estimate', 'Actual'])


# In[43]:


def try_parameters(w,b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    estimated_charges = estimate_charges(ages, w ,b)
    
    plt.plot(ages,estimated_charges,'r')
    sns.scatterplot(data = non_smoker_df , x='age' , y='charges' , alpha=0.8, s=100)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.title('Age VS Charges')
    plt.legend(['Estimate', 'Actual']) 




# In[44]:


try_parameters(250,-2020) #w changes the slope of the line and b moves the line up and down


# In[45]:


estimate_charges(40,250,-2020)


# In[46]:


targets = non_smoker_df.charges
targets


# In[47]:


predictions = estimated_charges
predictions


# In[48]:


def rmse(targets,predictions):
    return np.sqrt(np.mean(np.square(targets-predictions)))


# In[49]:


w=250
b=-1500


# In[50]:


try_parameters(w,b)


# In[51]:


targets = non_smoker_df['charges']
predictions = estimate_charges(non_smoker_df['age'] , w , b)


# In[52]:


rmse(targets , predictions)


# In[53]:


from sklearn.linear_model import LinearRegression


# In[54]:


model = LinearRegression()


# In[55]:


help(model.fit)


# In[56]:


inputs = non_smoker_df[['age']]
targets = non_smoker_df.charges
print('inputs.shape:' , inputs.shape)
print('inputs.targets:' , targets.shape)


# In[57]:


model.fit(inputs , targets)


# In[58]:


model.predict(np.array([[27],
                        [37],
                        [60]]))


# In[59]:


predictions = model.predict(inputs)


# In[60]:


predictions


# In[61]:


inputs


# In[62]:


targets


# In[63]:


rmse(targets,predictions)


# In[64]:


smoker_df = df[df['smoker'] == 'yes']


# In[65]:


plt.title('Age VS Charges')
sns.scatterplot(data = smoker_df , x='age' , y='charges' , alpha=0.8, s=100)


# In[66]:


def Estimated_charges(age, x , y):  # x=w   y=b
    return x*age + y


# In[67]:


x=800
y=100


# In[68]:


Estimated_charges(71 ,x ,y)


# In[69]:


Ages = smoker_df.age
Ages


# In[70]:


calc_charges = Estimated_charges(Ages , x, y)
calc_charges


# In[71]:


smoker_df.charges


# In[72]:


plt.plot(Ages,calc_charges,'r')
plt.xlabel('Ages')
plt.ylabel('Calculated_Charges')


# In[73]:


plt.plot(Ages,calc_charges,'r')
sns.scatterplot(data = smoker_df , x='age' , y='charges' , alpha=0.8, s=100)
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Age VS Charges')
plt.legend(['Calculated', 'Actual'])


# In[74]:


def Try_paramter(x,y):
    ages = smoker_df.age
    target = smoker_df.charges
    calc_charges = calc_charges(Ages, x ,y)
    
    
plt.plot(Ages,calc_charges,'r')
sns.scatterplot(data = smoker_df , x='age' , y='charges' , alpha=0.8, s=100)
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Age VS Charges')
plt.legend(['Calculated', 'Actual'])


# In[75]:


#Try_paramter(300,300)


# In[76]:


#giving the inputs
inputs , targets = non_smoker_df[['age']] , non_smoker_df[['charges']]

#training the model
model = LinearRegression().fit(inputs , targets)

#Generate the predictions
predictions = model.predict(inputs)

#Compute the Loss
loss = rmse(targets , predictions)
print('Loss: ', loss)


# In[77]:


#giving the inputs
Inputs , Targets = smoker_df[['age']] , smoker_df[['charges']]

#training the model
Model = LinearRegression().fit(Inputs , Targets)

#Generate the predictions
Predictions = model.predict(Inputs)

#Compute the Loss
Loss = rmse(Targets , Predictions)
print('Loss: ', Loss)


# In[78]:


model.coef_ , model.intercept_


# In[79]:


#giving the inputs
inputs , targets = df[['age' , 'bmi' , 'children']] , df[['charges']]

#training the model
Model = LinearRegression().fit(inputs , targets)

#Generate the predictions
Predictions = Model.predict(inputs)

#Compute the Loss
Loss = rmse(targets , Predictions)
print('Loss: ', Loss)


# In[80]:


sns.barplot(data = df , x ='smoker' , y ='charges')


# In[81]:


smoker_codes = {'no':0  , 'yes':1 }
df['smoker_codes'] = df.smoker.map(smoker_codes)


# In[82]:


df


# In[83]:


#giving the inputs
inputs , targets = df[['age' , 'bmi' , 'children' , 'smoker_codes']] , df[['charges']]

#training the model
Model = LinearRegression().fit(inputs , targets)

#Generate the predictions
Predictions = Model.predict(inputs)

#Compute the Loss
Loss = rmse(targets , Predictions)
print('Loss: ', Loss)


# In[84]:


sns.barplot(data = df , x ='sex' , y ='charges')


# In[85]:


gender_codes = {'female':0  , 'male':1 }
df['gender_codes'] = df.sex.map(gender_codes)


# In[86]:


df


# In[87]:


#giving the inputs
inputs , targets = df[['age' , 'bmi' , 'children' , 'smoker_codes' , 'gender_codes']] , df[['charges']]

#training the model
Model = LinearRegression().fit(inputs , targets)

#Generate the predictions
Predictions = Model.predict(inputs)

#Compute the Loss
Loss = rmse(targets , Predictions)
print('Loss: ', Loss)


# In[88]:


from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit(df[['region']])
enc.categories_


# In[89]:


enc.transform([['northeast']]).toarray()


# In[90]:


one_hot = enc.transform(df[['region']]).toarray()
one_hot


# In[91]:


df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot
df


# In[92]:


#giving the inputs
inputs , targets = df[['age' , 'bmi' , 'children' , 'smoker_codes' , 'gender_codes' , 'northeast' , 'northwest' , 'southeast' , 'southwest']] , df[['charges']]

#training the model
Model = LinearRegression().fit(inputs , targets)

#Generate the predictions
Predictions = Model.predict(inputs)

#Compute the Loss
Loss = rmse(targets , Predictions)
print('Loss: ', Loss)


# In[93]:


# FEATURE HANDLING
# To basically standardize the values
# for comapring every features' value to the model
#


# In[94]:


df


# In[95]:


from sklearn.preprocessing import StandardScaler 


# In[96]:


# Define the numeric columns to scale
numeric_cols = ['age', 'bmi', 'children']


# In[97]:


# Initialize the StandardScaler
scaler = StandardScaler()


# In[98]:


# Fit the scaler to the data
scaler.fit(df[numeric_cols])


# In[99]:


hasattr(scaler, 'mean_')


# In[100]:


df[numeric_cols].isnull().sum()


# In[101]:


scaler.mean_


# In[102]:


scaler.var_


# In[103]:


scaled_inputs = scaler.transform(df[numeric_cols])
scaled_inputs


# In[104]:


df


# In[105]:


cat_cols=['smoker_codes' , 'gender_codes' , 'northeast', 'northwest', 'southeast', 'southwest']
categorical_data = df[cat_cols].values
#categorical_data


# In[106]:


#giving the inputs
inputs , targets = np.concatenate((scaled_inputs , categorical_data), axis=1) , df[['charges']]

#training the model
Model = LinearRegression().fit(inputs , targets)

#Generate the predictions
Predictions = Model.predict(inputs)

#Compute the Loss
Loss = rmse(targets , Predictions)
print('Loss: ', Loss)


# In[107]:


df


# In[108]:


new_customers = [39 , 29 , 3 , 1 , 0 , 0 , 1 , 0 , 0]


# In[109]:


scaler.transform([[39 , 29 , 3]])


# In[110]:


Model.predict([[-0.01474046, -0.27287107,  1.580925761 , 0 , 0 , 1 , 0 , 0 ,0]])


# In[111]:


from sklearn.model_selection import train_test_split


# In[112]:


inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.1)


# In[113]:


# Create and train the model
model = LinearRegression().fit(inputs_train, targets_train)

# Generate predictions
predictions_test = model.predict(inputs_test)

# Compute loss to evalute the model
loss = rmse(targets_test, predictions_test)
print('Test Loss:', loss)


# In[114]:


# Generate predictions
predictions_train = model.predict(inputs_train)

# Compute loss to evalute the model
loss = rmse(targets_train, predictions_train)
print('Training Loss:', loss)


# In[ ]:




