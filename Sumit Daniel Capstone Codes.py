#Library section
import pandas as pd 
import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

pd.set_option('display.max_columns', 0) #this allow us to visualize all columns

#import the data attrition

import os

df = pd.read_csv("C:/Users/DELL/Desktop/python/capstone/imarticus projects/data/attrition.csv")
print(df.shape)

df.info() 
#check data type of all columns
#EDA

df.describe()

df['Attrition'].value_counts()
sns.countplot(df['Attrition'])


# histogram for age distribution
plt.figure(figsize=(10,8))
df['Age'].hist(bins=70)
plt.title("Age distribution of Employees")
plt.xlabel("Age")
plt.ylabel("Number of Employees")
plt.colorbar('red')
plt.show()

#checking attrition rate by monthly income

plt.figure(figsize=(14,10))
plt.scatter(df.Attrition,df.MonthlyIncome, alpha=.55)
plt.title("Attrition by Monthly Income ")
plt.ylabel("Monthly Income")
plt.grid(b=True, which='major',axis='y')
plt.show()

import matplotlib.pyplot as plt
fig_dims = (12, 4)
fig, ax = plt.subplots(figsize=fig_dims)
sns.countplot(x='Age', hue='Attrition', data = df, palette="colorblind")

#checking the  missing values

df.isna().sum() #Lets check if are missing values

df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

df['Over18'] = df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)
df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)

#Data Summary

df[df['Attrition'] == 1].describe()

df[df['Attrition'] == 0].describe()


#over time attrition
job_satisfaction = df.groupby(["OverTime", "Attrition"]).agg(count_col=pd.NamedAgg(column="Attrition", aggfunc="count")).reset_index()
fig = px.histogram(job_satisfaction, x="OverTime", y = 'count_col' ,color="Attrition")
fig.update_layout(barmode='group')
fig.show()

#feature selection
df.drop(columns = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber", "MonthlyRate", "DailyRate", "HourlyRate"], inplace = True)
df.shape

#input variables and output 
# Create an object scaler
MMS = MinMaxScaler()
# get dummies 
dummies = pd.get_dummies(df[df.columns.difference(["Attrition"])])
# scaling the data and define features
X = MMS.fit_transform(dummies)
# Define target variable
y = df[["Attrition"]].values.ravel()


#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0, shuffle = True)
Counter(y_train)

#modeling

#logistics

log_reg_model = LogisticRegression(max_iter=1000, solver = "newton-cg")
log_reg_model.fit(X_train, y_train)

y_pred = log_reg_model.predict(X_test)
print("Model accruracy score: {}".format(accuracy_score(y_test, y_pred)))

print(classification_report(y_test, y_pred))

#random forest classifier

random_forest_model = RandomForestClassifier(random_state = 0)
random_forest_model.fit(X_train, y_train)

y_pred = random_forest_model.predict(X_test)
print("Model accruracy score: {}".format(accuracy_score(y_test, y_pred)))

print(classification_report(y_test, y_pred))

#smote data

smt = SMOTE(random_state=0, sampling_strategy = 0.4)
X_train_SMOTE, y_train_SMOTE = smt.fit_sample(X_train, y_train)

Counter(y_train_SMOTE) #new shape of the target

#logistic regression on smote

log_reg_model = LogisticRegression(max_iter=1000, solver = "newton-cg")
log_reg_model.fit(X_train_SMOTE, y_train_SMOTE)

y_pred = log_reg_model.predict(X_test)
print("Model accruracy score: {}".format(accuracy_score(y_test, y_pred)))

print(classification_report(y_test, y_pred))

#random forest classifier and smote

random_forest_model = RandomForestClassifier(random_state = 0)
random_forest_model.fit(X_train_SMOTE, y_train_SMOTE)

y_pred = random_forest_model.predict(X_test)
print("Model accruracy score: {}".format(accuracy_score(y_test, y_pred)))

print(classification_report(y_test, y_pred))

