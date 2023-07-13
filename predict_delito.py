import sklearn
import sklearn.model_selection
from sklearn.metrics import accuracy_score
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
import pickle

#Preparamos el conjunto de datos que seran utilizados en el entrenamiento del modelo
dataset = pd.read_csv('dataset_2022_clasificacion.csv', sep=';', encoding='latin-1')
dataset = pd.get_dummies(dataset, columns=["categoria_hecho"], drop_first=False)
dataset = dataset.rename(columns={'categoria_hecho_DELITO DE BAJO IMPACTO': 'delito_bajo_impacto',
                                  'categoria_hecho_HECHO NO DELICTIVO': 'hecho_no_delictivo',
                                 'categoria_hecho_HOMICIDIO DOLOSO': 'homicidio_doloso',
                                 'categoria_hecho_LESIONES DOLOSAS POR DISPARO DE ARMA DE FUEGO':'lesiones_dolosas',
                                 'categoria_hecho_ROBO': 'robo',
                                 'categoria_hecho_SECUESTRO': 'secuestro',
                                 'categoria_hecho_VIOLACIÓN': 'violacion'})
#Seleccionamos los datos de test y entrenamiento
train, test = sklearn.model_selection.train_test_split(dataset, test_size=0.30, random_state=10)

#Calculamos el porcentaje de exito
def calculate_accuracy(model):
    delito_prediccion = model.predict(test) > 0.5
    print("Accuracy:", accuracy_score(test.delito, delito_prediccion))

#Entrenamos el modelo
model_all_features = smf.logit("delito ~ mes_hecho + dia_mes_hecho + dia_semana_hecho + hora + latitud + longitud", train).fit(method='bfgs')
calculate_accuracy(model_all_features)
predictions = model_all_features.predict(test) > 0.5

#Guardamos el entrenamiento
with open('predict_delito.pkl', 'wb') as predict:
    pickle.dump(model_all_features, predict)
