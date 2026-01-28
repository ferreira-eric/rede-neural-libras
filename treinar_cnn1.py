import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os



print("Carregando dataset")
try:
    df = pd.read_csv('dataset_libras.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'dataset_libras.csv' nao encontrado")
    exit()

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

num_amostras = X.shape[0]
if num_amostras == 0:
    print("Erro: Dataset vazio")
    exit()

X_reshaped = X.reshape(num_amostras, 21, 3)
print(f"Formato dos dados para CNN: {X_reshaped.shape}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
classes = encoder.classes_
np.save('classes.npy', classes)
print(f"Classes aprendidas: {classes}")

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_encoded, test_size=0.2, random_state=42
)

