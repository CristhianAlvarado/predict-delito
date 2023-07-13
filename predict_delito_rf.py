from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
import pickle

#Preparamos el conjunto de datos que seran utilizados en el entrenamiento del modelo
dataset = pd.read_csv('dataset_2022_clasificacion.csv', sep=';', encoding='latin-1')

#Seleccionamos los datos de test y entrenamiento
train, test = train_test_split(dataset, test_size=0.20, random_state=2)

#Entrenamos el modelo
model = RandomForestClassifier(n_estimators=10, random_state=1, verbose=False)

features = ["latitud", "longitud", "mes_hecho", "dia_mes_hecho", "dia_semana_hecho", "hora"]

model.fit(train[features], train.delito)

#Guardamos el entrenamiento
with open('predict_delito_rf.pkl', 'wb') as predict:
    pickle.dump(model, predict)
