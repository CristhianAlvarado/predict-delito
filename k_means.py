import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Función para realizar el agrupamiento K-means y visualizar los resultados
def kmeans_clustering(num_clusters):
    # Cargamos el dataset
    dataset = pd.read_csv('dataset_2022_clasificacion.csv', sep=';', encoding="latin-1")

    # Escalamiento de los datos
    scaler = StandardScaler()
    df_features = dataset[['latitud', 'longitud', 'hora']]
    X_scaled = scaler.fit_transform(df_features)
    
    # Construcción del modelo K-means
    kmeans = KMeans(n_clusters=num_clusters).fit(X_scaled)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    
    # Visualización de los resultados en 2D
    fig1, ax1 = plt.subplots()
    ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels)
    ax1.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', label='Centroides')
    ax1.set_xlabel('Latitud')
    ax1.set_ylabel('Longitud')
    ax1.set_title('Agrupación de Registros (2D)')
    ax1.legend()
    st.pyplot(fig1)
    # Interpretación
    st.write(f"Número de Clusters: {num_clusters}\n\n"
            "Los registros se han agrupado utilizando el algoritmo K-means en un gráfico de dispersión 2D. "
            "Cada punto representa un registro y está coloreado según el cluster al que pertenece. "
            "Los puntos rojos representan los centroides de los clusters. "
            "La posición en el gráfico representa la latitud y longitud, respectivamente.")
    
    # Visualización de los resultados en 3D
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=cluster_labels)
    ax2.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c='red', marker='X', s=100, label='Centroides')
    ax2.set_xlabel('Latitud')
    ax2.set_ylabel('Longitud')
    ax2.set_zlabel('Hora')
    ax2.set_title('Agrupación de Registros (3D)')
    ax2.legend()
    st.pyplot(fig2)
    # Interpretación
    st.write(f"Número de Clusters: {num_clusters}\n\n"
            "Los registros se han agrupado utilizando el algoritmo K-means en un gráfico de dispersión 3D. "
            "Cada punto representa un registro y está coloreado según el cluster al que pertenece. "
            "Los puntos rojos representan los centroides de los clusters. "
            "La posición en el gráfico representa la latitud, longitud y hora, respectivamente.")