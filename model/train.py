import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

#Load the dataset

data = fetch_california_housing(as_frame=True)
df = data.frame

#Preproces the dataset

X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal'] * 100000 

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

#Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

#Save the model
joblib.dump(model, 'model/california_housing.pkl')