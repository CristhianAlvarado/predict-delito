from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
import pickle

# Leemos los datos del csv
dataset = pd.read_csv('dataset_2022_clasificacion.csv', sep=';', encoding='latin-1')

# Elegimos los datos de entreno
train, test = train_test_split(dataset, test_size=0.30, random_state=2)

# Entrenamos el modelo
model = RandomForestClassifier(n_estimators=1, random_state=1, verbose=False)

features = ["latitud", "longitud", "mes_hecho", "dia_mes_hecho", "dia_semana_hecho", "hora", "delito"]

model.fit(train[features], train.categoria_hecho)

#Guardamos el entrenamiento
with open('predict_categoria_rf.pkl', 'wb') as predict:
    pickle.dump(model, predict)