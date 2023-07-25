from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
import pickle

# Leemos los datos del csv
dataset = pd.read_csv('dataset_2022_clasificacion.csv', sep=';', encoding='latin-1')

# Convertimos a dummies la columna categoria_hecho
dataset = pd.get_dummies(dataset, columns=["categoria_hecho"], drop_first=False)

# Renombramos las columnas
dataset = dataset.rename(columns={'categoria_hecho_DELITO DE BAJO IMPACTO': 'delito_bajo_impacto',
                                  'categoria_hecho_HECHO NO DELICTIVO': 'hecho_no_delictivo',
                                 'categoria_hecho_HOMICIDIO DOLOSO': 'homicidio_doloso',
                                 'categoria_hecho_LESIONES DOLOSAS POR DISPARO DE ARMA DE FUEGO':'lesiones_dolosas',
                                 'categoria_hecho_ROBO': 'robo',
                                 'categoria_hecho_SECUESTRO': 'secuestro',
                                 'categoria_hecho_VIOLACIÓN': 'violacion'})

# Elegimos los datos de entreno
train, test = train_test_split(dataset, test_size=0.30, random_state=2)

# Entrenamos el modelo
model = RandomForestClassifier(n_estimators=1, random_state=1, verbose=False)

# features = ["latitud", "longitud", "dia_mes_hecho", "dia_semana_hecho", "hora", "delito"]
features = ["delito_bajo_impacto", "robo", "lesiones_dolosas", "secuestro", "homicidio_doloso", "violacion", "hecho_no_delictivo", "dia_mes_hecho", "mes_hecho", "hora"]

model.fit(train[features], train.dia_semana_hecho)

#Guardamos el entrenamiento
with open('predict_dia_semana_rf.pkl', 'wb') as predict:
    pickle.dump(model, predict)