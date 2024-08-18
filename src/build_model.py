
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib

# load the dataset
df = pd.read_csv('../Data/Preprocessed_RedWineQuality.csv')

print(df.head())

# Prepare dataset for linear regression
x = df.drop(columns=['Quality']) # Feature
y = df['Quality'] # Targets

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=42)

# Initialising and training regression model
model = LinearRegression()
model.fit(x_train,y_train)

# Predicting on the test set
y_pred = model.predict(x_test)

#Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R^2): {r2}")

#dumping the model
joblib.dump(model, '../model/LinearRegressionModel.pkl')
