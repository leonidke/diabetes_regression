#Import libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


#load the dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target


#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Create linear regression object
model = LinearRegression()


#Train the model using the training sets
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#The coefficients
print('Coefficients:', model.coef_)
print('The intercept is:', model.intercept_)

#The mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)


#The coefficient of determination: 1 is the perfect prediction
r2 = r2_score(y_test, y_pred)
print('Coefficient of Determination:', r2)

# Plot outputs
plt.scatter(y_test, y_pred, color='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linewidth=3)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Measured vs Predicted')
plt.show()

