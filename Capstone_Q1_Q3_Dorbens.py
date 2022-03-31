#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import tensorflow as tf
from tensorflow import keras
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot
import numpy as np
import os
import datetime
from tensorflow.python.data import Dataset
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,multilabel_confusion_matrix,mean_squared_error,recall_score,precision_score,f1_score,ConfusionMatrixDisplay


# In[2]:


df_q1= pd.read_csv('Data/columnALL.csv', index_col=False)


# In[3]:


print(len(df_q1))
print(len(df_q1.columns))


# In[5]:


df_q1.drop(columns='Unnamed: 0', axis=0, inplace=True)
columnAll = df_q1.copy()
df_q1= pd.get_dummies(df_q1, columns=['home_ownership','application_type','purpose','addr_state',], drop_first=True)
df_q1=pd.get_dummies(df_q1, columns=['loan_status'])
df_q1['average_fico'] = df_q1[['fico_range_low','fico_range_high']].mean(axis=1)
df_q1.drop(['fico_range_low','fico_range_high'], axis=1, inplace=True)


# In[6]:


df_q1.isna().sum().sort_values(ascending=False)


# In[7]:


df_q1.fillna(0,inplace=True)
columnAll.fillna(0,inplace=True)


# In[8]:


df_q1.isna().sum().sort_values(ascending=False).head(20)


# In[9]:


from matplotlib import pyplot as plt
plt.figure(figsize = (10, 6), dpi = 100)
plt.subplot(1, 2,1) # row 1, col 2 index 1
plt.bar(columnAll.loan_status.unique(),columnAll.loan_status.value_counts(),color = ['green','red','yellow'])
for index,data in enumerate(columnAll.loan_status.value_counts()):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=16),ha='center')
plt.title("Status",  fontweight='bold')
plt.xticks([])
plt.xlabel('Loan Status', fontweight='bold', labelpad =10)
plt.ylabel('Total Amount in Millions',  fontweight='bold', labelpad=10)


plt.subplot(1, 2,2) # index 2
_, _, autopcts = plt.pie(columnAll.loan_status.value_counts(),autopct='%1.2f%%',
        wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'}, explode = (0, 0.3, 0.3), radius=1.5, colors = ['green','red','yellow'])
plt.setp(autopcts, **{'color':'black', 'weight':'bold', 'fontsize':12.5}, ha='left')
plt.axis('equal')
plt.legend(columnAll.loan_status.unique().tolist())
plt.show()
plt.savefig('loan_status_count.png')


# In[10]:


def over_corr(df):
    df_corr = df.corr(method='pearson')
    df_unstack = df_corr.unstack()
    corr_sorted = df_unstack.sort_values(kind="quicksort")
    x = corr_sorted[(corr_sorted > .75) & (corr_sorted<1)]
    x = x.drop_duplicates()
    over_corr_q1 = []
    for x in x.keys():
      for y in x:
       over_corr_q1.append(y)
    over_corr_q1 = list(set(over_corr_q1))
    b=df.corr(method='pearson').loc[over_corr_q1]
    
    return b.iloc[:,-4:-1]


# In[11]:


over_corr(df_q1)


# In[12]:


removed_over_corr = ['application_type_Joint App','num_op_rev_tl','installment','int_rate','bc_util', 'num_actv_bc_tl','total_pymnt']
df_q1.drop(columns=removed_over_corr, inplace=True)

df_corr = df_q1.corr(method='pearson')
df_unstack = df_corr.unstack()
corr_sorted = df_unstack.sort_values(kind="quicksort")
x = corr_sorted[(corr_sorted > .75) & (corr_sorted<1)]
x = x.drop_duplicates()
x
# In[13]:


over_corr(df_q1)


# In[14]:


removed_over_corr_2 = ['num_sats','num_tl_30dpd', 'num_bc_tl' ]
df_q1.drop(columns=removed_over_corr_2, inplace=True)


# In[15]:


over_corr(df_q1)


# In[16]:


features = df_q1.drop(columns=['loan_status_dpi','loan_status_Dafault','loan_status_Late','loan_status_Paid'])


# In[17]:


df_q1.head()


# In[18]:


#Split the data
x_train_q1,x_test_q1,y_train_q1,y_test_q1 = train_test_split(features, df_q1.iloc[:,-4:-1], test_size=0.20, random_state=42, stratify=df_q1.iloc[:,-4:-1],shuffle=True)


# In[19]:



Dtclassifier = DecisionTreeClassifier().fit(x_train_q1, y_train_q1)
y_pred_q1= Dtclassifier.predict(x_test_q1)


# In[20]:


def confusion_scores(y_test, y_pred):
    print("Accuracy:",accuracy_score(y_test, y_pred))
    print("Precision:",precision_score(y_test, y_pred,average = 'macro'))
    print("Recall:",recall_score(y_test, y_pred,average = 'macro'))
    print("F1 Score:",f1_score(y_test, y_pred,average = 'macro'))


# In[21]:


confusion_scores(y_test_q1, y_pred_q1)


# In[22]:


multi_con_mat = multilabel_confusion_matrix(y_test_q1, y_pred_q1)
print('Classification Report:\n', classification_report(y_test_q1, y_pred_q1,target_names=['Loan Defaulted','Loan Late','Loan Paid']))


# In[23]:


train_scores, test_scores = list(), list()
# define the tree depths to evaluate
values = [i for i in range(1, 21)]
# evaluate a decision tree for each depth
for i in values:
	# configure the model
	model = DecisionTreeClassifier(max_depth=i)
	# fit model on the training dataset
	model.fit(x_train_q1, y_train_q1)
	# evaluate on the train dataset
	train_yhat = model.predict(x_train_q1)
	train_acc = accuracy_score(y_train_q1, train_yhat)
	train_scores.append(train_acc)
	# evaluate on the test dataset
	test_yhat = model.predict(x_test_q1)
	test_acc = accuracy_score(y_test_q1, test_yhat)
	test_scores.append(test_acc)
	# summarize progress
	print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
# plot of train and test scores vs tree depth
pyplot.plot(values, train_scores, '-o', label='Train')
pyplot.plot(values, test_scores, '-o', label='Test')
pyplot.legend()
pyplot.show()

#Question 3: Predict States affect on Defaulting
# In[22]:


df_q3= features.copy()


# In[23]:


loan_state = pd.read_csv('Data/columnBin.csv', index_col=False)
df_q3['loan_status_state']=loan_state['loan_status_state']


# In[24]:


#replace missing values with a 0
df_q3.replace(np.nan, 0, inplace=True)
#checking for any missing values
df_q3[df_q3.isnull().any(axis=1)]


# In[25]:


df_q3 = df_q3.loc[: , ~df_q3.columns.str.contains('addr_state')]


# In[26]:



#df_q3.drop(labels=[480895],axis=0,inplace=True)
#df_q3.loan_status_state = LabelEncoder().fit_transform(df_q3.loan_status_state)
df_q3.head()


# In[27]:


#Contains one label for the y target
df_q3.drop(labels=[1787832],axis=0,inplace=True)


# In[28]:


df_q3 = df_q3[df_q3.loan_status_state!=37]


# In[29]:


x_train_q3, x_test_q3, y_train_q3, y_test_q3 = train_test_split(df_q3.iloc[:,:-1],df_q3.iloc[:,-1:],test_size=.2,random_state=1, stratify=df_q3.iloc[:,-1:])


# In[ ]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
svm_model = LinearSVC()
svm_model.fit(x_train_q3,y_train_q3)


# In[31]:


df_q3.loan_status_state = LabelEncoder().fit_transform(df_q3.loan_status_state)


# In[34]:


get_ipython().system('pip install imbalanced-learn')


# In[ ]:


from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE, ADASYN
smen = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
adas = ADASYN()
x_train_smeen, y_train_smeen =smen.fit_resample(x_train_q3,y_train_q3)


# In[86]:


len(x_train_q3)


# In[28]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
nb_model = OneVsRestClassifier(GaussianNB())
nb_model.fit(x_train_q3,y_train_q3)


# In[29]:


y_pred_q3 = nb_model.predict(x_test_q3)


# In[30]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test_q3, y_pred_q3))


# In[46]:


metrics.multilabel_confusion_matrix(y_test_q3, y_pred_q3)


# In[ ]:




