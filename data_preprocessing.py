# Ideal Pytorch Pipeline - Data Preparation

#importing necessary libraries
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# loading the dataset
df = pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')
print(df.head())

print(df.shape)

#Dropping the unneeded columns id and unnamed:32 and modifying the original dataset with inplace
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)
print(df.head())

# Train-test splitting
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.2)

# show a sample of X_train like in the notebook
print(X_train)

# Scaling
#WE want to scale the dataset so we have normalized values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#use of fit and transform expalined in label encoding

print(X_train)

print(y_train)

# Label Encoding
#we use label encoding because we can see that labels are represented using alphabets M,B
# changing these to 0 and 1 which our NN can understand

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

#using fit creates universal mappings for the entirety of dataset and further transforms will use same mappings
# fit assigns using alphabetical order so benign=0,malig=1

print(encoder.classes_)
print(y_train)

