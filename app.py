import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import phik
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import phate
from FRUFS import FRUFS

st.set_page_config(page_title="Feature Engineering & EDA", layout="wide")

st.title("Plataforma de Feature Engineering y EDA")
st.markdown("Prototipo de análisis para reducción de dimensionalidad.")

st.sidebar.header("1. Carga de Archivos")
uploaded_files = st.sidebar.file_uploader("Sube tus archivos CSV", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    dfs = [pd.read_csv(file) for file in uploaded_files]
    df = pd.concat(dfs, ignore_index=True)
    
    st.write(f"### Datos Cargados (Total filas: {df.shape[0]}, Columnas: {df.shape[1]})")
    st.dataframe(df.head())

    st.header("2. Análisis Exploratorio de Datos (EDA)")
    if st.button("Generar Mapa de Correlación Phi_K"):
        with st.spinner("Calculando..."):
            phik_matrix = df.phik_matrix()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(phik_matrix, annot=True, cmap="coolwarm")
            st.pyplot(fig)

    st.header("3. Reducción de Dimensionalidad")
    n_comp = st.slider("Componentes a extraer", 2, 6, 3)
    
    if st.button("Ejecutar PCA y UMAP"):
        with st.spinner("Procesando algoritmos..."):
            pca = PCA(n_components=n_comp)
            pca_feat = pca.fit_transform(df)
            for i in range(n_comp): df[f'PCA_{i+1}'] = pca_feat[:, i]
            
            reducer = umap.UMAP(n_components=n_comp, random_state=42)
            umap_feat = reducer.fit_transform(df.iloc[:, :-n_comp])
            for i in range(n_comp): df[f'UMAP_{i+1}'] = umap_feat[:, i]
            
            st.success("Características generadas.")
            st.dataframe(df.head())
else:
    st.info("Sube un archivo CSV en el panel izquierdo para iniciar.")
