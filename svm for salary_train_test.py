# -*- coding: utf-8 -*-
"""
Created on Wed May 25 22:51:31 2022

@author: Sai pranay
"""

#-----------------------------imporing the data set---------------------------

import pandas as pd

train = pd.read_csv("E:\\DATA_SCIENCE_ASS\\SUPPORT VECTOR MACHINE\\SalaryData_Train(1).csv")
print(train)
list(train)
train.shape
train.describe()
train.info()
train.dtypes
train.hist()


test = pd.read_csv("E:\\DATA_SCIENCE_ASS\\SUPPORT VECTOR MACHINE\\SalaryData_Test(1).csv")
print(test)
list(test)
test.shape
test.describe()
test.info()
test.dtypes
test.hist()

#---------------------------- checking correlation-----------------------------

train.corr()
test.corr()

#-------------------plot------

import seaborn as sns

sns.pairplot(test)
sns.pairplot(train)

#---------------------------label encoding------------------------------------

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()


train["workclass"] = lb.fit_transform(train["workclass"])
train["education"] = lb.fit_transform(train["education"])
train["maritalstatus"] = lb.fit_transform(train["maritalstatus"])
train["occupation"] = lb.fit_transform(train["occupation"])
train["relationship"] = lb.fit_transform(train["relationship"])
train["race"] = lb.fit_transform(train["race"])
train["sex"] = lb.fit_transform(train["sex"])
train["native"] = lb.fit_transform(train["native"])
train["Salary"] = lb.fit_transform(train["Salary"])


test["workclass"] = lb.fit_transform(test["workclass"])
test["education"] = lb.fit_transform(test["education"])
test["maritalstatus"] = lb.fit_transform(test["maritalstatus"])
test["occupation"] = lb.fit_transform(test["occupation"])
test["relationship"] = lb.fit_transform(test["relationship"])
test["race"] = lb.fit_transform(test["race"])
test["sex"] = lb.fit_transform(test["sex"])
test["native"] = lb.fit_transform(test["native"])
test["Salary"] = lb.fit_transform(test["Salary"])

#-----------------------------spitting the data set----------------------------

X_train=train.iloc[:,:-1]
X_train

y_train=train.iloc[:,-1]
y_train

X_test=test.iloc[:,:-1]
X_test

y_test = test.iloc[:,-1]
y_test

X_train.shape
y_train.shape
X_test.shape
y_test.shape


#---------------------------model fitting--------------------------------------

from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)

#----------------------------Predicting model----------------------------------

y_pred = model.predict(X_test)
y_pred

#-----------------------------Model Evaluation---------------------------------
from sklearn.metrics import confusion_matrix, classification_report


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



from sklearn import metrics

cm_test = metrics.confusion_matrix(y_test, y_pred)
print(cm_test)

print("Testing Accuracy:",metrics.accuracy_score(y_test, y_pred).round(2))

#------------------------------------------------------------------------------

# Training a classifier - kernel='rbf'

from sklearn.svm import SVC
SVC()
# clf = SVC(kernel='linear')
#clf = SVC(kernel='poly',degree=3)
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred1=clf.predict(X_test)
y_pred1

from sklearn import metrics

cm_test = metrics.confusion_matrix(y_test, y_pred1)
print(cm_test)

print("Testing Accuracy:",metrics.accuracy_score(y_test, y_pred1).round(2))

#------------------------------------------------------------------------------
# Training a classifier - kernel='linear'


from sklearn.svm import SVC
SVC()
clf = SVC(kernel='linear')
#clf = SVC(kernel='poly',degree=3)
#clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred2=clf.predict(X_test)
y_pred2

from sklearn import metrics

cm_test = metrics.confusion_matrix(y_test, y_pred2)
print(cm_test)

print("Testing Accuracy:",metrics.accuracy_score(y_test, y_pred2).round(2))

