import streamlit as st
import pydeck as pdk

# Función para mostrar el mapa y obtener la ubicación seleccionada


def mostrar_mapa():
    st.title("Seleccionar ubicación en el mapa")

    # Mostrar el mapa en Streamlit utilizando st.pydeck_chart
    mapa = st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=0,
            longitude=0,
            zoom=2,
            pitch=0,
        ),
    ))

    # Obtener las coordenadas (latitud y longitud) de la ubicación seleccionada al hacer clic
    if st.button("Obtener Coordenadas"):
        if mapa is not None:
            click_info = st.pydeck_chart.get_click_info(mapa)
            if click_info:
                latitud = click_info["lat"]
                longitud = click_info["lon"]

                # Mostrar las coordenadas en un cuadro de texto
                st.write("Latitud:", latitud)
                st.write("Longitud:", longitud)


if __name__ == "__main__":
    mostrar_mapa()
