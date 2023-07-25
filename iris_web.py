#importar librerias
import streamlit as st
import pickle
import pandas as pd
from k_means import kmeans_clustering

#Datos para los inputs
meses = {
    "Enero": 1,
    "Febrero": 2,
    "Marzo": 3,
    "Abril": 4,
    "Mayo": 5,
    "Junio": 6,
    "Julio": 7,
    "Agosto": 8,
    "Septiembre": 9,
    "Octubre": 10,
    "Noviembre": 11,
    "Diciembre": 12
}

dias_semana = {
    "Lunes": 1,
    "Martes": 2,
    "Miércoles": 3,
    "Jueves": 4,
    "Viernes": 5,
    "Sábado": 6,
    "Domingo": 7
}

horas_hecho = {
    "Entre 6 am y 9 am": 1,
    "Entre 9 am y 12 pm": 2,
    "Entre 12 pm y 3 pm": 3,
    "Entre 3 pm y 6 pm": 4,
    "Entre 6 pm y 9 pm": 5,
    "Entre 9 pm y 12 am": 6,
    "Entre 0 am y 3 am": 7,
    "Entre 3 am y 6 am": 8
}

categorias = {
    "Delito bajo impacto": 1,
    "Hecho no delictivo": 2,
    "Homicidio doloso": 3,
    "Lesiones dolosas por disparo": 4,
    "Robo": 5,
    "Secuestro": 6,
    "Violación": 7
}

#Extraer los archivos pickles
with open('predict_delito_rf.pkl', 'rb') as predict_delito:
    predict_delito_rf = pickle.load(predict_delito)

with open('predict_mes_rf.pkl', 'rb') as predict_mes:
    predict_mes_rf = pickle.load(predict_mes)

with open('predict_categoria_rf.pkl', 'rb') as predict_categoria:
    predict_categoria_rf = pickle.load(predict_categoria)

with open('predict_dia_semana_rf.pkl', 'rb') as predict_dia_semana:
    predict_dia_semana_rf = pickle.load(predict_dia_semana)


def classify_delito(num):
    print(num.item())
    if num.item() > 0.5:
        return 'Ocurre un delito'
    else:
        return 'No ocurre un delito'
    
def classify_mes(num):
    return next((mes for mes, index in meses.items() if index == num), None)

def classify_categoria(cat):
    print(cat)
    return 'Se trata de un ' + cat.item()

def classify_dia_semana(num):
    print(num)
    return next((dia for dia, index in dias_semana.items() if index == num), None)
      

def page_one():
    df = input_predict()

    #Escoger el modelo preferido
    st.write(df)

    if st.button('RUN'):
        st.success(classify_delito(predict_delito_rf.predict(df)))

def page_two():
    df = input_mes()

    st.write(df)

    if st.button('RUN'):
        st.success("Es probable que el delito ocurra en el mes de " + classify_mes(predict_mes_rf.predict(df)))

def page_three():
    df = input_categoria()
    
    st.write(df)

    if st.button('RUN'):
        st.success(classify_categoria(predict_categoria_rf.predict(df)))

def page_four():
    # Crear la aplicación Streamlit
    st.title('Aprendizaje No Supervisado - Algoritmo K-Means')

    # Menú para seleccionar el número de clusters con input range (2 a 5)
    num_clusters = st.sidebar.slider('Número de Clusters:', min_value=2, max_value=5, value=4)

    # Mostrar el gráfico de agrupamiento
    kmeans_clustering(num_clusters)

def page_five():
    df = input_dia_semana()

    st.write(df)

    if st.button('RUN'):
        st.success(classify_dia_semana(predict_dia_semana_rf.predict(df)))


def input_predict():
    latitud = st.sidebar.text_input('Latitud', 19.44427347, placeholder="Latitud")
    longitud = st.sidebar.text_input('Longitud', -99.08785745, placeholder="Longitud")
    mes_hecho = st.sidebar.selectbox('Mes', options=list(meses.keys()))
    dia_mes_hecho = st.sidebar.slider('Día de mes', 1, 31, 5)
    dia_semana_hecho = st.sidebar.selectbox('Día de semana', options=list(dias_semana.keys()))
    hora_hecho = st.sidebar.radio('Hora del hecho', options=list(horas_hecho.keys()))

    data_map = pd.DataFrame({
        'lat': [float(latitud)],
        'lon': [float(longitud)],
    })

    st.map(data_map)

    data = {
        'latitud': float(latitud),
        'longitud': float(longitud),
        'mes_hecho': meses[mes_hecho],
        'dia_mes_hecho': dia_mes_hecho,
        'dia_semana_hecho': dias_semana[dia_semana_hecho],
        'hora': horas_hecho[hora_hecho]
    }

    features = pd.DataFrame(data, index=[0])
    return features

def input_mes():
    # categoria = st.sidebar.selectbox('Categoria delito', options=list(categorias.keys()))
    latitud = st.sidebar.text_input('Latitud', 19.44427347, placeholder="Latitud")
    longitud = st.sidebar.text_input('Longitud', -99.08785745, placeholder="Longitud")
    dia_mes_hecho = st.sidebar.slider('Día de mes', 1, 31, 5)
    dia_semana_hecho = st.sidebar.selectbox('Día de semana', options=list(dias_semana.keys()))
    hora_hecho = st.sidebar.radio('Hora del hecho', options=list(horas_hecho.keys()))
    
    data_map = pd.DataFrame({
        'lat': [float(latitud)],
        'lon': [float(longitud)],
    })

    st.map(data_map)

    data = {
        'latitud': float(latitud),
        'longitud': float(longitud),
        'dia_mes_hecho': dia_mes_hecho,
        'dia_semana_hecho': dias_semana[dia_semana_hecho],
        'hora': horas_hecho[hora_hecho],
        "delito": 1
    }

    features = pd.DataFrame(data, index=[0])
    return features

def input_categoria():
    latitud = st.sidebar.text_input('Latitud', 19.44427347, placeholder="Latitud")
    longitud = st.sidebar.text_input('Longitud', -99.08785745, placeholder="Longitud")
    dia_mes_hecho = st.sidebar.slider('Día de mes', 1, 31, 5)
    mes_hecho = st.sidebar.selectbox('Mes', options=list(meses.keys()))
    dia_semana_hecho = st.sidebar.selectbox('Día de semana', options=list(dias_semana.keys()))
    hora_hecho = st.sidebar.radio('Hora del hecho', options=list(horas_hecho.keys()))
    
    data_map = pd.DataFrame({
        'lat': [float(latitud)],
        'lon': [float(longitud)],
    })

    st.map(data_map)

    data = {
        'latitud': float(latitud),
        'longitud': float(longitud),
        'mes_hecho': meses[mes_hecho],
        'dia_mes_hecho': dia_mes_hecho,
        'dia_semana_hecho': dias_semana[dia_semana_hecho],
        'hora': horas_hecho[hora_hecho],
        "delito": 1
    }

    features = pd.DataFrame(data, index=[0])
    return features

def input_dia_semana():
    categoria = st.sidebar.selectbox('Categoria delito', options=list(categorias.keys()))
    dia_mes_hecho = st.sidebar.slider('Día de mes', 1, 31, 5)
    mes_hecho = st.sidebar.selectbox('Mes', options=list(meses.keys()))
    hora_hecho = st.sidebar.radio('Hora del hecho', options=list(horas_hecho.keys()))
    
    data = {
        'delito_bajo_impacto': 1 if categorias[categoria] == 1 else 0,
        'robo': 1 if categorias[categoria] == 5 else 0,
        'lesiones_dolosas': 1 if categorias[categoria] == 4 else 0,
        'secuestro': 1 if categorias[categoria] == 6 else 0,
        'homicidio_doloso': 1 if categorias[categoria] == 3 else 0,
        'violacion': 1 if categorias[categoria] == 7 else 0,
        'hecho_no_delictivo': 1 if categorias[categoria] == 2 else 0,
        'dia_mes_hecho': dia_mes_hecho,
        'mes_hecho': meses[mes_hecho],
        'hora': horas_hecho[hora_hecho]
    }
    
    features = pd.DataFrame(data, index=[0])
    return features


def main():
    # Opciones para la navegación
    pages = {
        "Predecir delito": page_one,
        "Predecir mes": page_two,
        "Predecir categoria": page_three,
        "Predecir dia de semana": page_five,
        "Algoritmo K-means": page_four
    }

    st.title('Modelamiento de hechos delictivos')
    st.sidebar.header('User Input Parameters')
    st.subheader('User Input Parameters')

    # Radio button para seleccionar la página
    page_selection = st.radio("Selecciona un modelo:", tuple(pages.keys()))

    # Ejecutar la función correspondiente a la página seleccionada
    if page_selection in pages:
        pages[page_selection]()

if __name__ == '__main__':
    main()