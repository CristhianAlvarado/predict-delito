#importar librerias
import streamlit as st
import folium
from streamlit_folium import folium_static
import pickle
import pandas as pd

#Extraer los archivos pickles
with open('predict_delito.pkl', 'rb') as predict:
    predict_delito = pickle.load(predict)

with open('predict_delito_rf.pkl', 'rb') as predict_rf:
    predict_delito_rf = pickle.load(predict_rf)

# with open('linear_reg.pkl', 'rb') as li:
#     lin_reg = pickle.load(li)

# with open('log_reg.pkl', 'rb') as lo:
#     log_reg = pickle.load(lo)

# with open('svc_m.pkl', 'rb') as sv:
#     svc_m = pickle.load(sv)


def classify(num):
    print(num.item())
    if num.item() > 0.5:
        return 'Ocurre un delito'
    else:
        return 'No ocurre un delito'

def main():
    st.title('Modelamiento de hechos delictivos')
    st.sidebar.header('User Input Parameters')
    st.subheader('User Input Parameters')
    
    def user_input_parameters():
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

        latitud= st.sidebar.text_input('Latitud', 19.44427347, placeholder="Latitud")
        longitud = st.sidebar.text_input('Longitud', -99.08785745, placeholder="Longitud")
        mes_hecho = st.sidebar.selectbox('Mes', options=list(meses.keys()))
        dia_mes_hecho = st.sidebar.slider('Día de mes', 1, 31, 5)
        dia_semana_hecho = st.sidebar.selectbox('Día de semana', options=list(dias_semana.keys()))
        hora_hecho = st.sidebar.radio('Hora del hecho', options=list(horas_hecho.keys()))

        data = pd.DataFrame({
            'lat': [float(latitud)],
            'lon': [float(longitud)],
        })

        st.map(data)

        data = {
            'latitud': float(latitud),
            'longitud': float(longitud),
            'mes_hecho': meses[mes_hecho],
            'dia_mes_hecho': dia_mes_hecho,
            'dia_semana_hecho': dias_semana[dia_semana_hecho],
            'hora': horas_hecho[hora_hecho],
        }

        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_parameters()

    #Escoger el modelo preferido
    option = ['Logistic Regression', 'Random Forest']
    model = st.sidebar.selectbox('¿Qué modelo deseas usar?', option)

    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'Logistic Regression':
            st.success(classify(predict_delito_rf.predict(df)))

if __name__ == '__main__':
    main()