# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:55:30 2022

@author: Sai pranay
"""
#------------------------------importing the data set--------------------------
import pandas as pd
ff = pd.read_csv("E:\\DATA_SCIENCE_ASS\\SUPPORT VECTOR MACHINE\\forestfires.csv")
print(ff)
ff.shape
ff.head()
ff.info()
ff.describe()
list(ff)



#------------------------------checking_for_null_values------------------------

ff.isnull().sum()

#-------------------------------label_encoding---------------------------------

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
ff["month_n"] = LE.fit_transform(ff["month"])
ff[["month","month_n"]].head(11)
pd.crosstab(ff.month,ff.month_n)

ff["day_n"] = LE.fit_transform(ff["day"])
ff[["day","day_n"]].head(11)
pd.crosstab(ff.day,ff.day_n)

ff["size_category_n"] = LE.fit_transform(ff["size_category"])
ff[["size_category","size_category_n"]].head(11)
pd.crosstab(ff.size_category,ff.size_category_n)

#-----------------------rearranging_the_dataset--------------------------------

ff_new = ff.drop(['month','day','size_category'],axis = 1)
ff_new.shape
list(ff_new)

ff_2 = ff_new.iloc[:,:28]
ff_2.shape
list(ff_2)


ff_3 = ff_new.iloc[:,28:]
print(ff_3)






#---------------------standardization_the_data_set-----------------------------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scale = scaler.fit_transform(ff_2)
X_scale

X_new1 = pd.DataFrame(X_scale)
X_new1


#------------------------------------------------------------------------------



ff_4 = pd.concat([X_new1, ff_3],axis = 1)
print(ff_4)
ff_4.shape
ff_4.head()
ff_4.info()
ff_4.describe()



#-------------------------checking_the_correlation_point-----------------------

import matplotlib.pyplot as plt
import seaborn as sns

ff_4.corr()

plt.figure(figsize=(5,5))
sns.heatmap(ff_4,annot=True)

#======================spitting_the_data---------------------------------------

x = ff_4.iloc[:,:30]
print(x)
list(x)
x.shape

y = ff_4.iloc[:,30:]
print(y)
list(y)
y.shape
y.ndim


from sklearn.model_selection._split import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=10)
x_train.shape
x_test.shape

y_train.shape
y_test.shape
y_test.ndim

#---------------------svm_model_deployment-------------------------------------

# Training a classifier - kernel='rbf'

from sklearn.svm import SVC
SVC()
# clf = SVC(kernel='linear')
#clf = SVC(kernel='poly',degree=3)
clf = SVC(kernel='rbf')
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)


from sklearn import metrics

cm_test = metrics.confusion_matrix(y_test, y_pred)
print(cm_test)



print("Testing Accuracy:",metrics.accuracy_score(y_test, y_pred).round(2))


#---------------------svm_best_model_deployment-------------------------------------


from sklearn.svm import SVC
SVC()

clf = SVC(kernel='linear')
# clf = SVC(kernel='poly',degree=3)
#clf = SVC(kernel='rbf')

clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)


from sklearn import metrics

cm_test = metrics.confusion_matrix(y_test, y_pred)
print(cm_test)

print("Testing Accuracy:",metrics.accuracy_score(y_test, y_pred).round(2))
