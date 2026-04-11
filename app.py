import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import phik
from sklearn.decomposition import PCA
import umap

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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Ver Serie de Tiempo (CO y NOx)"):
            with st.spinner("Graficando emisiones..."):
                fig, axes = plt.subplots(2, 1, figsize=(10, 6))
                
                # Gráfica CO
                axes[0].scatter(df.index, df['CO'], s=2, color='black')
                axes[0].set_ylabel('CO values')
                
                # Gráfica NOx
                axes[1].scatter(df.index, df['NOX'], s=2, color='black')
                axes[1].set_ylabel('NOx values')
                axes[1].set_xlabel('sample')
                
                plt.tight_layout()
                st.pyplot(fig)

    with col2:
        if st.button("Generar Pairplot (KDE)"):
            with st.spinner("Generando Pairplot... Esto toma un par de minutos."):
                # Tomamos una muestra aleatoria de 1000 datos para que el servidor gratuito no colapse dibujando millones de puntos
                df_sample = df.sample(n=min(1000, len(df)), random_state=42)
                fig = sns.pairplot(df_sample, diag_kind="kde", markers=".", height=1.5)
                st.pyplot(fig)

    with col3:
        if st.button("Generar Correlación Phi_K"):
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
            
            st.success("Características generadas. Revisa las últimas columnas de la tabla.")
            st.dataframe(df.head())
else:
    st.info("Sube un archivo CSV en el panel izquierdo para iniciar.")
