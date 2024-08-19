

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping# type: ignore
import random

#adding seed value
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

df= pd.read_csv('../data/Preprocessed_RedWineQuality.csv')

# Preparing dataset for model building

x = df.drop(columns='Quality')
y = df['Quality']





# Adding normalisation layer
normalization_layer = tf.keras.layers.Normalization(axis=-1)
normalization_layer.adapt(x.to_numpy())


# split the data into Training,cross-validation and testing

x_train_val,x_test,y_train_val,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
x_train,x_cv,y_train,y_cv = train_test_split(x_train_val,y_train_val,test_size=0.25,random_state=42)


input_size = x_train.shape[1]
model = Sequential([
    
    normalization_layer,
     Dense(400, input_shape=(input_size,), activation='relu'),
    Dense(200, activation='sigmoid'),
    Dense(100, activation='relu'),
    Dense(50,activation='sigmoid'),
    Dense(25,activation='relu'),
    Dense(1,activation='relu')

])
model.compile(optimizer ='adam',
              loss = 'mse',
              metrics = ['mae']
              )
# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

history = model.fit(x_train, y_train, epochs=50, batch_size=32,validation_data=(x_cv,y_cv),callbacks=[early_stopping])



joblib.dump(model,'../model/neuralnetwork.pkl')

