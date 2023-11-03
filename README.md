# Programming-with-Python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

##Training & Testing Dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
ideal = pd.read_csv('ideal.csv')

train.head()
ideal.head()
test.head()
test.shape

# Statistical Summary of data
##Statistical Summary of data
train.describe()

##Information of training data
train.info()
test.info()

##Checking Null values
train.isnull().sum()
test.isnull().sum()

#Exploratory Data Analysis
##EDA
import matplotlib.pyplot as plt
plt.scatter(train['x'], train['y4'])
plt.title("x Vs y4")
plt.xlabel('x')
plt.ylabel('y4')
plt.show()

plt.scatter(train['x'], train['y3'])
plt.title('x Vs y3')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.scatter(train['x'], train['y4'])
plt.title('x Vs y4')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

##Distribution Plot
import seaborn as sns
sns.displot(train['y2'])
sns.displot(train['y3'])
sns.boxplot(train['y4'])

Correlation
##Correlation Plot
import seaborn as sns
plt.figure(figsize=(20,15))
correlation = train.corr()
sns.heatmap(correlation, annot=True)
Preparing Dataset for Training and Testing
##Traning and Testing data spliting
X = test.drop('y', axis=1)
y = test['y']
y.shape
X.shape

Regression Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

Linear Regression

###Linear Regression Model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)
print('Coefficient of model :', lin_reg_model.coef_)
print('Intercept of model :',lin_reg_model.intercept_)

# Root Mean Squared Error on training dataset

predict_train = lin_reg_model.predict(X_train)
rmse_train = mean_squared_error(y_train,predict_train)**(0.5)
print('\nRMSE on train dataset : ', rmse_train)

## prediction on test data splitting from metadata
predict_test = lin_reg_model.predict(X_test)

rmse_test = mean_squared_error(y_test,predict_test)**(0.5)

print('\nRMSE on test dataset : ', rmse_test)
