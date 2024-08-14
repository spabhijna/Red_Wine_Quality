
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib

# load the dataset
df = pd.read_csv('/Volumes/Code/RedWineQuality/Data/Preprocessed_RedWineQuality.csv')

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

# Plotting predicted vs actual values
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred, alpha=0.7,edgecolors='k')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max(),],'r--', lw=2)
plt.xlabel('Actual quality')
plt.ylabel('Predicted quality')
plt.title('Actual vs Predicted wine quality')
plt.show()

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7, edgecolors='k')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Quality')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Quality')
plt.show()

# Histogram of residual
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

# Get feature names and coefficients
coefficients = model.coef_
features = x.columns

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x=coefficients, y=features)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Coefficients')
plt.show()


train_sizes, train_scores, test_scores = learning_curve(
    model, x, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend(loc='best')
plt.show()


joblib.dump(model, '/Volumes/Code/RedWineQuality/model/LinearRegressionModel.pkl')
