import pandas as pd
import numpy as np 
import ast
import os
import requests
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import folium
from streamlit.components.v1 import html
import joblib
import io

# --- DESCARGAR ARCHIVOS DESDE GITHUB SI NO EXISTEN ---
ARCHIVOS = {
    "Modelo_P_h.h5": "https://github.com/Ezecordoba/Consultora_ProData/raw/main/MODELOS/Modelo_P_h.h5",
    "Modelo_P_C.h5": "https://github.com/Ezecordoba/Consultora_ProData/raw/main/MODELOS/Modelo_P_C.h5",
    "modelo_xgb_1.pkl": "https://github.com/Ezecordoba/Consultora_ProData/raw/main/MODELOS/modelo_xgb_1.pkl",
    "metadatos_ML.csv": "https://github.com/Ezecordoba/Consultora_ProData/raw/main/MODELOS/metadatos_ML.csv",
    "atributos.csv": "https://github.com/Ezecordoba/Consultora_ProData/raw/main/MODELOS/atributos.csv",
    "categorias.csv": "https://github.com/Ezecordoba/Consultora_ProData/raw/main/MODELOS/categorias.csv",
    "ciudad_categoria_procesado.csv": "https://github.com/Ezecordoba/Consultora_ProData/raw/main/MODELOS/ciudad_categoria_procesado.csv"
}

def descargar_archivo(nombre, url, binario=False):
    if not os.path.isfile(nombre):
        r = requests.get(url)
        if r.status_code == 200:
            with open(nombre, "wb" if binario else "w") as f:
                f.write(r.content if binario else r.text)
        else:
            st.error(f"No se pudo descargar {nombre}")

# Descargar modelos y datos
for archivo, url in ARCHIVOS.items():
    descargar_archivo(archivo, url, binario=archivo.endswith((".h5", ".pkl")))

# Cargar modelos
modelo_P_h = load_model("Modelo_P_h.h5")
modelo_P_h.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])

modelo_P_C = load_model("Modelo_P_C.h5")
modelo_P_C.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])

modelo_xgb = joblib.load("modelo_xgb_1.pkl")

# Cargar datos
def cargar_csv_desde_url(url):
    r = requests.get(url)
    return pd.read_csv(io.BytesIO(r.content))

metadatos1 = cargar_csv_desde_url(ARCHIVOS["metadatos_ML.csv"])
atributos = cargar_csv_desde_url(ARCHIVOS["atributos.csv"])
categorias = cargar_csv_desde_url(ARCHIVOS["categorias.csv"])
ciudad_categoria = cargar_csv_desde_url(ARCHIVOS["ciudad_categoria_procesado.csv"])


# --- FUNCIONES DE CADA PÁGINA ---

def atributos_destacados(df, atributos):
    df['atributo_nombres'] = df['atributo_id'].apply(
        lambda x: [atributos[atributos['atributo_id'] == id_]['atributo'].values[0] for id_ in x])
    todos_los_atributos = df['atributo_nombres'].explode().tolist()
    frecuencia_atributos = pd.Series(todos_los_atributos).value_counts()
    return frecuencia_atributos.index.tolist()

def pagina_recomendar_ciudades():
    st.title("🌍 Recomendación de ciudades por categoría de restaurante")

    categoria = st.selectbox('Selecciona una categoría:', [''] + categorias['category'].tolist())

    if categoria:
        try:
            categoria = categoria.lower()
            categorias["category"] = categorias["category"].str.lower()
            metadatos2 = metadatos1.copy()

            id_categoria = categorias[categorias["category"] == categoria]
            id_categoria = id_categoria.reset_index(drop=True)

            metadatos2['category_id'] = metadatos2['category_id'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            metadatos2['atributo_id'] = metadatos2['atributo_id'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

            df_categ = metadatos2[metadatos2['category_id'].apply(lambda x: id_categoria.loc[0, "category_id"] in x)]
            df_categ = df_categ.reset_index(drop=True)

            numero_restaurantes = df_categ.groupby('city_id').size()
            df_categ["numero_restaurantes"] = df_categ['city_id'].map(numero_restaurantes)
            df_categ['category_id'] = id_categoria.loc[0, "category_id"]

            scaler = MinMaxScaler()
            X_c = df_categ[['reviews_positivas', 'reviews_negativas', 'stars']]
            X_c = scaler.fit_transform(X_c)
            y_c = modelo_P_C.predict(X_c)
            df_categ["(1/P_C)_i"] = y_c
            df_categ["1/P_C"] = df_categ.groupby('city_id')["(1/P_C)_i"].transform('sum') / df_categ["numero_restaurantes"]

            df_categ_ciu = df_categ.drop_duplicates(subset='city_id', ignore_index=True)
            X_h = df_categ_ciu[['population', 'numero_restaurantes']]
            X_h = scaler.fit_transform(X_h)
            y_h = modelo_P_h.predict(X_h)
            df_categ_ciu["P_h"] = y_h

            a = 2.5
            b = 1
            df_categ_ciu["Phi"] = a * df_categ_ciu["P_h"] + b * df_categ["1/P_C"]

            df_categ_ciu = df_categ_ciu.sort_values(by='Phi', ascending=False)
            top_ciudades = df_categ_ciu[["city", "latitude", "longitude"]].head(3).reset_index(drop=True)

            st.subheader(f'Top 3 ciudades para la categoría "{categoria.title()}"')
            st.dataframe(top_ciudades["city"])

            map_florida = folium.Map(location=[27.9944024, -81.7602544], zoom_start=7)
            for _, row in top_ciudades.iterrows():
                folium.Marker([row['latitude'], row['longitude']], popup=row['city']).add_to(map_florida)

            st.subheader("Mapa de ciudades recomendadas")
            map_html = map_florida._repr_html_()
            st.components.v1.html(map_html, height=500)

            df_categ = df_categ.sort_values(by='1/P_C', ascending=True)
            top_atributos = atributos_destacados(df_categ, atributos)
            top10_atributos = pd.DataFrame(top_atributos[:10], columns=["Atributos"])
            st.subheader("Top 10 atributos destacados")
            st.write(top10_atributos)

        except Exception as e:
            st.error(f"Ocurrió un error: {str(e)}")

def pagina_recomendar_categoria():
    st.title("🏙️ Recomendación de categorías por ciudad")

    ciudad_usuario = st.text_input("Ingresa la ciudad:")

    if st.button("Predecir"):
        if ciudad_usuario:
            df_ciudad = ciudad_categoria[ciudad_categoria["city"] == ciudad_usuario][[
                "city", "category", "competencia", "avg_rating", "avg_vader_score", "avg_textblob_score", "poblacion"
            ]]

            if df_ciudad.empty:
                st.warning(f"No hay datos disponibles para la ciudad: {ciudad_usuario}")
                return

            X = df_ciudad[["competencia", "avg_rating", "avg_vader_score", "avg_textblob_score", "poblacion"]]
            df_ciudad["recomendado"] = modelo_xgb.predict(X)
            recomendadas = df_ciudad[df_ciudad["recomendado"] == 1]["category"]

            if recomendadas.empty:
                st.info("No hay categorías recomendadas.")
            else:
                st.success("Categorías recomendadas:")
                st.write(recomendadas.to_frame().reset_index(drop=True))
        else:
            st.error("Por favor, ingresa una ciudad.")

# --- NAVEGACIÓN ENTRE PÁGINAS ---
st.sidebar.title("📂 Navegación")
pagina = st.sidebar.radio("Selecciona una página", (
    "Ciudades recomendadas por categoría", 
    "Categorías recomendadas por ciudad"
))

if pagina == "Ciudades recomendadas por categoría":
    pagina_recomendar_ciudades()
else:
    pagina_recomendar_categoria()

