import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy import stats

# read the dataset 
dataset_path = '/Volumes/Code/RedWineQuality/Data/Red Wine Quality.csv'
df = pd.read_csv(dataset_path)

# Display first few cells
print(df.head())

# Basic Understanding of Data
print(f"Dataset info:\n {df.info()}")
print(f"Summary stats:\n {df.describe()}")

# Checking for missing values
print(f'total missing values are:\n{df.isnull().sum()}')
# Data has no missing values

# Checking for Outliers using boxplot 

'''plt.figure(figsize=(15,12))
df.boxplot(color=dict(boxes='red', whiskers='purple', medians='blue', caps='blue'))
plt.xticks(rotation = 90)
plt.title('Boxplot of dataset')
plt.show()'''

# Melting df to use it in seaborn
df_melted = df.melt(var_name='Variable', value_name='Value')

# Generating  a box plot 
plt.figure(figsize=(15, 12))
sns.boxplot(x='Variable', y='Value', data=df_melted, color='yellow')
plt.xticks(rotation=90)
plt.title('Boxplot of Dataset')
plt.show()

# Handling outliers
z_score = np.abs(stats.zscore(df.select_dtypes(include=np.number)))
df_clean = df[(z_score < 3).all(axis=1)]
print(f"\nNumber of rows after outlier removal: {df_clean.shape[0]} (original: {df.shape[0]})")

# Feature Engineering 
df_clean = df_clean.copy() # Ensure df_clean is a copy to avoid warnings
df_clean.loc[:, 'total acidity'] = df_clean['fixed acidity'] + df_clean['volatile acidity']
df_clean.loc[:, 'sulphates_squared'] = df_clean['sulphates'] ** 2
df_clean.loc[:, 'citric_acid_ratio'] = df_clean['citric acid'] / df_clean['fixed acidity']
df_clean.loc[:, 'sugar_acid_diff'] = df_clean['residual sugar'] - df_clean['fixed acidity']
df_clean.loc[:, 'Quality'] = df_clean['quality']

# Drop the original column
df_clean = df_clean.drop('quality', axis=1)

print(df_clean)

# Generate scatter plots for all features against the 'quality'
for i, column in enumerate(df_clean.columns[:-1]): # to exclude quality column
    plt.subplot(5,3,i+1)
    sns.scatterplot(x=df_clean[column],y=df_clean['Quality'])
    plt.title(f'Scatter plot of {column} v/s Quality')
    plt.xlabel(column)
    plt.ylabel('Quality')

plt.tight_layout()
plt.show()

# Analysing variables affecting quality of wine
coor_mat = df_clean.corr()
plt.figure(figsize=(15,12))
sns.heatmap(data=coor_mat, annot=True,cmap='coolwarm',fmt ='.2f')
plt.title('Correlation Matrix')
plt.show()

# To save the preprossed data for building a model
df_clean.to_csv('/Volumes/Code/RedWineQuality/Data/Preprocessed_RedWineQuality.csv')







