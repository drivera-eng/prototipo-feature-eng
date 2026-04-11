import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import phik
from sklearn.decomposition import PCA
import umap
import numpy as np

st.set_page_config(page_title="Feature Engineering & EDA Dinámico", layout="wide")

st.title("Plataforma de Feature Engineering y EDA")
st.markdown("Prototipo de análisis universal para cualquier conjunto de datos.")

# --- INICIALIZACIÓN DE LA MEMORIA ---
if 'show_timeseries' not in st.session_state:
    st.session_state.show_timeseries = False
if 'show_pairplot' not in st.session_state:
    st.session_state.show_pairplot = False
if 'show_correlation' not in st.session_state:
    st.session_state.show_correlation = False
if 'show_pca_scatter' not in st.session_state:
    st.session_state.show_pca_scatter = False
if 'show_umap_scatter' not in st.session_state:
    st.session_state.show_umap_scatter = False
if 'df_with_features' not in st.session_state:
    st.session_state.df_with_features = None

# 1. Carga de Archivos
st.sidebar.header("1. Carga de Archivos")
uploaded_files = st.sidebar.file_uploader("Sube tus archivos CSV", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    dfs = [pd.read_csv(file) for file in uploaded_files]
    df_raw = pd.concat(dfs, ignore_index=True)
    
    # IMPORTANTE: Filtrar solo columnas numéricas para evitar errores matemáticos con texto
    df_numeric = df_raw.select_dtypes(include=[np.number]).dropna()
    
    if st.session_state.df_with_features is not None:
        df = st.session_state.df_with_features
    else:
        df = df_numeric

    st.write(f"### Datos Cargados (Total filas: {df.shape[0]}, Columnas Numéricas: {df.shape[1]})")
    st.dataframe(df.head())

    st.header("2. Análisis Exploratorio de Datos (EDA)")
    
    # --- BOTONES DE EDA ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Selección dinámica de columnas para la serie de tiempo
        st.markdown("**Serie de Tiempo**")
        selected_cols = st.multiselect("Elige las variables a graficar:", df_numeric.columns.tolist(), default=df_numeric.columns.tolist()[:2])
        if st.button("Generar Serie de Tiempo"):
            st.session_state.show_timeseries = True
            st.session_state.ts_cols = selected_cols

    with col2:
        st.markdown("**Distribución de Datos**")
        if st.button("Generar Pairplot (KDE)"):
            st.session_state.show_pairplot = True

    with col3:
        st.markdown("**Matriz de Dependencias**")
        if st.button("Generar Correlación Phi_K"):
            st.session_state.show_correlation = True

    # --- RENDERIZADO DE GRÁFICAS DE EDA ---
    
    # Serie de Tiempo Dinámica
    if st.session_state.show_timeseries and 'ts_cols' in st.session_state:
        if len(st.session_state.ts_cols) > 0:
            with st.expander("📈 Series de Tiempo", expanded=True):
                with st.spinner("Graficando..."):
                    fig, axes = plt.subplots(len(st.session_state.ts_cols), 1, figsize=(10, 3 * len(st.session_state.ts_cols)))
                    # Ajustar si es solo una gráfica o múltiples
                    if len(st.session_state.ts_cols) == 1:
                        axes = [axes]
                        
                    for ax, col in zip(axes, st.session_state.ts_cols):
                        ax.scatter(df.index, df[col], s=2, color='black')
                        ax.set_ylabel(col)
                    
                    axes[-1].set_xlabel('Index / Sample')
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.warning("Selecciona al menos una variable para graficar.")

    # Pairplot
    if st.session_state.show_pairplot:
        with st.expander("📊 Pairplot con Densidad de Kernel (KDE)", expanded=True):
            with st.spinner("Generando Pairplot (usando muestra aleatoria para velocidad)..."):
                # Limitar variables si hay demasiadas para que no colapse visualmente (max 10)
                cols_to_plot = df_numeric.columns[:10]
                df_sample = df[cols_to_plot].sample(n=min(1000, len(df)), random_state=42)
                fig = sns.pairplot(df_sample, diag_kind="kde", markers=".", height=1.5)
                st.pyplot(fig)

    # Correlación
    if st.session_state.show_correlation:
        with st.expander("🔥 Mapa de Correlación Phi_K", expanded=True):
            with st.spinner("Calculando..."):
                phik_matrix = df_numeric.phik_matrix()
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(phik_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 8})
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

    st.header("3. Reducción de Dimensionalidad")
    n_comp = st.slider("Componentes a extraer (PCA y UMAP)", 2, 6, 3)
    
    col1_rd, col2_rd = st.columns(2)
    
    with col1_rd:
        if st.button("Ejecutar Cálculos (PCA y UMAP)"):
            with st.spinner("Procesando algoritmos en la nube..."):
                pca = PCA(n_components=n_comp)
                pca_feat = pca.fit_transform(df_numeric)
                for i in range(n_comp): df_numeric[f'PCA_{i+1}'] = pca_feat[:, i]
                
                reducer = umap.UMAP(n_components=n_comp, random_state=42)
                umap_feat = reducer.fit_transform(df_numeric.iloc[:, :-n_comp])
                for i in range(n_comp): df_numeric[f'UMAP_{i+1}'] = umap_feat[:, i]
                
                st.session_state.df_with_features = df_numeric
                st.success(f"Características extraídas con éxito.")
                st.dataframe(df_numeric.head())

    with col2_rd:
        if st.button("Ver Gráficos de Dispersión (2D)"):
            if st.session_state.df_with_features is not None:
                st.session_state.show_pca_scatter = True
                st.session_state.show_umap_scatter = True
            else:
                st.error("Primero ejecuta los cálculos.")

    if st.session_state.show_pca_scatter and st.session_state.show_umap_scatter:
        df_feat = st.session_state.df_with_features
        with st.expander("Visualizaciones 2D de Componentes", expanded=True):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            sns.scatterplot(data=df_feat, x='PCA_1', y='PCA_2', color='blue', s=10, ax=ax1)
            ax1.set_title("PCA: 2 Primeros Componentes")
            sns.scatterplot(data=df_feat, x='UMAP_1', y='UMAP_2', color='orange', s=10, ax=ax2)
            ax2.set_title("UMAP: 2 Primeros Componentes")
            st.pyplot(fig)

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Limpiar Todo y Reiniciar"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()

else:
    st.info("Sube un archivo CSV en el panel izquierdo para iniciar.")
