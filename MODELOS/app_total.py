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
    "trained_model.h5": "https://github.com/Ezecordoba/Consultora_ProData/raw/main/MODELOS/trained_model.h5",
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

for archivo, url in ARCHIVOS.items():
    descargar_archivo(archivo, url, binario=archivo.endswith((".h5", ".pkl")))

modelo_P_h = load_model("Modelo_P_h.h5")
modelo_P_h.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])

modelo_P_C = load_model("Modelo_P_C.h5")
modelo_P_C.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mse'])

modelo_xgb = joblib.load("modelo_xgb_1.pkl")
modelo_caracteristicas = load_model("trained_model.h5")

def cargar_csv_desde_url(url):
    r = requests.get(url)
    return pd.read_csv(io.BytesIO(r.content))

metadatos1 = cargar_csv_desde_url(ARCHIVOS["metadatos_ML.csv"])
atributos = cargar_csv_desde_url(ARCHIVOS["atributos.csv"])
categorias = cargar_csv_desde_url(ARCHIVOS["categorias.csv"])
ciudad_categoria = cargar_csv_desde_url(ARCHIVOS["ciudad_categoria_procesado.csv"])

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
            id_categoria = categorias[categorias["category"] == categoria].reset_index(drop=True)
            metadatos2['category_id'] = metadatos2['category_id'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            metadatos2['atributo_id'] = metadatos2['atributo_id'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df_categ = metadatos2[metadatos2['category_id'].apply(lambda x: id_categoria.loc[0, "category_id"] in x)].reset_index(drop=True)
            numero_restaurantes = df_categ.groupby('city_id').size()
            df_categ["numero_restaurantes"] = df_categ['city_id'].map(numero_restaurantes)
            df_categ['category_id'] = id_categoria.loc[0, "category_id"]
            scaler = MinMaxScaler()
            X_c = scaler.fit_transform(df_categ[['reviews_positivas', 'reviews_negativas', 'stars']])
            y_c = modelo_P_C.predict(X_c)
            df_categ["(1/P_C)_i"] = y_c
            df_categ["1/P_C"] = df_categ.groupby('city_id')["(1/P_C)_i"].transform('sum') / df_categ["numero_restaurantes"]
            df_categ_ciu = df_categ.drop_duplicates(subset='city_id', ignore_index=True)
            X_h = scaler.fit_transform(df_categ_ciu[['population', 'numero_restaurantes']])
            y_h = modelo_P_h.predict(X_h)
            df_categ_ciu["P_h"] = y_h
            df_categ_ciu["Phi"] = 2.5 * df_categ_ciu["P_h"] + df_categ["1/P_C"]
            df_categ_ciu = df_categ_ciu.sort_values(by='Phi', ascending=False)
            top_ciudades = df_categ_ciu[["city", "latitude", "longitude"]].head(3).reset_index(drop=True)
            st.subheader(f'Top 3 ciudades para la categoría "{categoria.title()}"')
            st.dataframe(top_ciudades["city"])
            map_florida = folium.Map(location=[27.9944024, -81.7602544], zoom_start=7)
            for _, row in top_ciudades.iterrows():
                folium.Marker([row['latitude'], row['longitude']], popup=row['city']).add_to(map_florida)
            st.subheader("Mapa de ciudades recomendadas")
            st.components.v1.html(map_florida._repr_html_(), height=500)
            df_categ = df_categ.sort_values(by='1/P_C', ascending=True)
            top10_atributos = pd.DataFrame(atributos_destacados(df_categ, atributos)[:10], columns=["Atributos"])
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

def pagina_prediccion_caracteristicas():
    st.title("🏪 Predicción por características del negocio")
    st.subheader("Modelo de Selección de Características")
    st.write("Elegí las opciones que querés ofrecer en tu negocio. El modelo predecirá si tendrá éxito.")
    caracteristicas_esp = [
        'delivery', 'para llevar', 'comer en el local', 'asientos al aire libre', 'autoservicio', 'bueno para trabajar con laptop',
        'cenas en solitario', 'accesible para sillas de ruedas', 'bebidas alcohólicas', 'comida saludable', 'comida rápida confort',
        'menú en braille', 'todo lo que puedas comer', 'café', 'baile', 'servicio de catering', 'servicio en mostrador',
        'pago por adelantado', 'asientos', 'desayuno', 'almuerzo', 'cena', 'postre', 'casual', 'romántico', 'formal', 'moderno',
        'con reservación', 'suele haber espera', 'visita rápida', 'liderado por personas de color', 'liderado por mujeres',
        'liderado por veteranos', 'entretenimiento', 'espectáculos en vivo', 'amigable con LGBTQ+', 'servicio rápido',
        'chimenea', 'asientos en azotea', 'deportes', 'para estudiantes universitarios', 'familiar', 'grupos', 'lugareños',
        'turistas', 'amigable para niños', 'wifi', 'bar en el lugar', 'solo efectivo', 'cheques', 'tarjetas de crédito',
        'tarjetas de débito', 'pagos móviles NFC', 'reciclaje'
    ]
    input_data = np.zeros((1, 54))
    col1, col2, col3 = st.columns(3)
    for i, caracteristica in enumerate(caracteristicas_esp):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.checkbox(caracteristica):
                input_data[0, i] = 1
    prediction = modelo_caracteristicas.predict(input_data)
    prediction = np.round(prediction, 1)
    st.markdown("### 🔮 Resultado de predicción:")
    if prediction == 1:
        st.success("✅ El modelo predice que el negocio tiene **éxito**.")
    else:
        st.error("❌ El modelo predice que el negocio **no tendría éxito**.")

# --- NAVEGACIÓN ENTRE PÁGINAS ---
st.sidebar.title("📂 Navegación")
pagina = st.sidebar.radio("Selecciona una página", (
    "Ciudades recomendadas por categoría",
    "Categorías recomendadas por ciudad",
    "Predicción por características del negocio"
))

if pagina == "Ciudades recomendadas por categoría":
    pagina_recomendar_ciudades()
elif pagina == "Categorías recomendadas por ciudad":
    pagina_recomendar_categoria()
elif pagina == "Predicción por características del negocio":
    pagina_prediccion_caracteristicas()


