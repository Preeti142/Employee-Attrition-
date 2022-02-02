
# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds=pd.read_excel("data.xls")
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
LE= LabelEncoder()
ds["Attrition"]=LE.fit_transform(ds["Attrition"])
ds["BusinessTravel"]=LE.fit_transform(ds["BusinessTravel"])
ds["Department"]=LE.fit_transform(ds["Department"])
ds["EducationField"]=LE.fit_transform(ds["EducationField"])
ds["Gender"]=LE.fit_transform(ds["Gender"])
ds["JobRole"]=LE.fit_transform(ds["JobRole"])
ds["MaritalStatus"]=LE.fit_transform(ds["MaritalStatus"])
ds["OverTime"]=LE.fit_transform(ds["OverTime"])
X= ds.drop(["Attrition"],axis=1)
y= ds["Attrition"]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set # without Considering balance
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression( random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import accuracy_score,confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
accuracy_sc=accuracy_score(y_test,y_pred)
print(accuracy_sc)

#roc curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
log_roc_auc1 = roc_auc_score(y_test, y_pred)
print(log_roc_auc1)
#plot roc
plt.figure()
plt.plot(fpr, tpr, label = 'Logistic regression(area = %0.2f)' %log_roc_auc1)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operating Charecteristic')
plt.legend(loc = "lower right")
plt.show()


#Feature Elimination

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
X=ds.drop(["Attrition","HourlyRate"],axis=1)
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
g=print(result.summary2())
X=ds.drop(["Attrition","HourlyRate","BusinessTravel"],axis=1)
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
g=print(result.summary2())
X=ds.drop(["Attrition","HourlyRate","BusinessTravel","Education"],axis=1)
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
g=print(result.summary2())
X=ds.drop(["Attrition","HourlyRate","BusinessTravel","Education","MonthlyIncome"],axis=1)
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
g=print(result.summary2())
X=ds.drop(["Attrition","HourlyRate","BusinessTravel","Education","MonthlyIncome","MonthlyRate"],axis=1)
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
g=print(result.summary2())
X=ds.drop(["Attrition","HourlyRate","BusinessTravel","Education","MonthlyIncome","MonthlyRate","EducationField"],axis=1)
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
g=print(result.summary2())
X=ds.drop(["Attrition","HourlyRate","BusinessTravel","Education","MonthlyIncome","MonthlyRate","EducationField","DailyRate"],axis=1)
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
g=print(result.summary2())
X=ds.drop(["Attrition","HourlyRate","BusinessTravel","Education","MonthlyIncome","MonthlyRate","EducationField","DailyRate","StockOptionLevel"],axis=1)
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
g=print(result.summary2())
X=ds.drop(["Attrition","HourlyRate","BusinessTravel","Education","MonthlyIncome","MonthlyRate","EducationField","DailyRate","StockOptionLevel","JobRole"],axis=1)
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
g=print(result.summary2())

X=ds.drop(["Attrition","HourlyRate","BusinessTravel","Education","MonthlyIncome","MonthlyRate","EducationField","DailyRate","StockOptionLevel","JobRole","TrainingTimesLastYear"],axis=1)
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
print(X.columns)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Logistic Regression to the Training set #Considering balance
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(class_weight = 'balanced', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import accuracy_score,confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
cm1

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred1))
accuracy_sc1=accuracy_score(y_test,y_pred1)
print(accuracy_sc1)
#roc curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])

log_roc_auc1 = roc_auc_score(y_test, y_pred1)
print(log_roc_auc1)
#plot roc
plt.figure()
plt.plot(fpr, tpr, label = 'Logistic regression(area = %0.2f)' %log_roc_auc1)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operating Charecteristic')
plt.legend(loc = "lower right")
plt.show()


