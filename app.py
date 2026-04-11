from scipy.stats import norm
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import phik
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import umap
import numpy as np

st.set_page_config(page_title="Feature Engineering & EDA Dinámico", layout="wide")

st.title("Plataforma de Feature Engineering y EDA")
st.markdown("Prototipo de análisis universal con protección de memoria y laboratorio interactivo.")

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
# Actualizamos los tipos aceptados
uploaded_files = st.sidebar.file_uploader(
    "Sube tus archivos (CSV, Excel u ODS)", 
    type=["csv", "xlsx", "ods"], 
    accept_multiple_files=True
)

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        # Identificamos el tipo de archivo por su extensión
        if file.name.endswith('.csv'):
            dfs.append(pd.read_csv(file))
        elif file.name.endswith('.xlsx'):
            dfs.append(pd.read_excel(file, engine='openpyxl'))
        elif file.name.endswith('.ods'):
            dfs.append(pd.read_excel(file, engine='odf'))
    
    df_raw = pd.concat(dfs, ignore_index=True)
    df_numeric = df_raw.select_dtypes(include=[np.number]).dropna()
    
    # --- VÁLVULA DE SEGURIDAD (ANTI-CRASH) ---
    MAX_ROWS = 5000
    if len(df_numeric) > MAX_ROWS:
        st.warning(f"⚠️ Dataset gigante detectado ({len(df_numeric)} filas). Se ha extraído una muestra de {MAX_ROWS} filas.")
        df_numeric = df_numeric.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
    
    if st.session_state.df_with_features is not None:
        df = st.session_state.df_with_features
    else:
        df = df_numeric

    st.write(f"### Datos Activos (Filas en memoria: {df.shape[0]}, Columnas: {df.shape[1]})")
    st.dataframe(df.head())

    st.header("2. Análisis Exploratorio de Datos (EDA)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Serie de Tiempo**")
        selected_cols = st.multiselect("Elige las variables:", df_numeric.columns.tolist(), default=df_numeric.columns.tolist()[:2])
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

    # --- RENDERIZADO EDA ---
    if st.session_state.show_timeseries and 'ts_cols' in st.session_state:
        if len(st.session_state.ts_cols) > 0:
            with st.expander("📈 Series de Tiempo", expanded=True):
                fig, axes = plt.subplots(len(st.session_state.ts_cols), 1, figsize=(10, 3 * len(st.session_state.ts_cols)))
                if len(st.session_state.ts_cols) == 1: axes = [axes]
                for ax, col in zip(axes, st.session_state.ts_cols):
                    ax.scatter(df.index, df[col], s=2, color='black')
                    ax.set_ylabel(col)
                axes[-1].set_xlabel('Index / Sample')
                plt.tight_layout()
                st.pyplot(fig)

    if st.session_state.show_pairplot:
        with st.expander("📊 Pairplot con KDE", expanded=True):
            cols_to_plot = df_numeric.columns[:10]
            fig = sns.pairplot(df[cols_to_plot], diag_kind="kde", markers=".", height=1.5)
            st.pyplot(fig)

    if st.session_state.show_correlation:
        with st.expander("🔥 Mapa de Correlación Phi_K", expanded=True):
            phik_matrix = df_numeric.phik_matrix()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(phik_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

    st.header("3. Reducción de Dimensionalidad")
    n_comp = st.slider("Componentes a extraer", 2, 6, 3)
    
    col1_rd, col2_rd = st.columns(2)
    
    with col1_rd:
        if st.button("Ejecutar Cálculos (PCA y UMAP)"):
            pca = PCA(n_components=n_comp)
            pca_feat = pca.fit_transform(df_numeric)
            for i in range(n_comp): df_numeric[f'PCA_{i+1}'] = pca_feat[:, i]
            
            reducer = umap.UMAP(n_components=n_comp, random_state=42)
            umap_feat = reducer.fit_transform(df_numeric.iloc[:, :-n_comp])
            for i in range(n_comp): df_numeric[f'UMAP_{i+1}'] = umap_feat[:, i]
            
            st.session_state.df_with_features = df_numeric
            st.success("Características extraídas.")
            st.dataframe(df_numeric.head())

    with col2_rd:
        if st.button("Ver Gráficos de Dispersión"):
            if st.session_state.df_with_features is not None:
                st.session_state.show_pca_scatter = True
                st.session_state.show_umap_scatter = True
            else:
                st.error("Primero ejecuta los cálculos.")

    if st.session_state.show_pca_scatter and st.session_state.show_umap_scatter:
        df_feat = st.session_state.df_with_features
        with st.expander("Visualizaciones 2D", expanded=True):
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
        st.rerun()

else:
    st.info("Sube un archivo (CSV, Excel u ODS) para iniciar.")

st.markdown("---")

# 4. LABORATORIO INTERACTIVO
st.header("4. Laboratorio Interactivo: Modelos y Distribuciones")
if uploaded_files and not df_numeric.empty:
    tab1, tab2 = st.tabs(["📈 Regresión Polinómica", "🔔 Distribución de Gauss"])
    
    with tab1:
        st.markdown("### Ajuste de Modelos Predictivos")
        col_x, col_y = st.columns(2)
        with col_x:
            var_x = st.selectbox("Eje X:", df_numeric.columns.tolist(), index=0, key="x_reg")
        with col_y:
            idx_y = 1 if len(df_numeric.columns) > 1 else 0
            var_y = st.selectbox("Eje Y:", df_numeric.columns.tolist(), index=idx_y, key="y_reg")

        grado = st.slider("Grado Polinómico", 1, 15, 1)

        df_plot = df_numeric.sample(n=min(500, len(df_numeric)), random_state=42).sort_values(by=var_x)
        X_real = df_plot[[var_x]].values
        y_real = df_plot[var_y].values

        poly = PolynomialFeatures(degree=grado, include_bias=False)
        X_poly = poly.fit_transform(X_real)
        model = LinearRegression().fit(X_poly, y_real)

        X_seq = np.linspace(X_real.min(), X_real.max(), 300).reshape(-1, 1)
        y_seq = model.predict(poly.transform(X_seq))

        fig_lab, ax_lab = plt.subplots(figsize=(10, 5))
        sns.scatterplot(x=X_real.ravel(), y=y_real, color="black", alpha=0.6, s=40, ax=ax_lab)
        ax_lab.plot(X_seq, y_seq, color="red", linewidth=2.5, label=f"Grado {grado}")
        ax_lab.set_xlabel(var_x); ax_lab.set_ylabel(var_y); ax_lab.legend()
        st.pyplot(fig_lab)

    with tab2:
        st.markdown("### Análisis de Frecuencias")
        var_gauss = st.selectbox("Variable:", df_numeric.columns.tolist(), index=0, key="var_gauss")
        bins = st.slider("Intervalos", 10, 100, 30)
        
        data_gauss = df_numeric[var_gauss].dropna()
        mu, std = norm.fit(data_gauss)
        fig_gauss, ax_gauss = plt.subplots(figsize=(10, 5))
        sns.histplot(data_gauss, bins=bins, stat="density", color="skyblue", alpha=0.6, ax=ax_gauss)
        
        xmin, xmax = ax_gauss.get_xlim()
        x_curve = np.linspace(xmin, xmax, 200)
        ax_gauss.plot(x_curve, norm.pdf(x_curve, mu, std), 'darkorange', linewidth=2.5, label=f"Gauss: $\mu={mu:.2f}, \sigma={std:.2f}$")
        ax_gauss.set_title(f"Distribución de {var_gauss}"); ax_gauss.legend()
        st.pyplot(fig_gauss)
