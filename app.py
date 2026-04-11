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
uploaded_files = st.sidebar.file_uploader("Sube tus archivos CSV", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    dfs = [pd.read_csv(file) for file in uploaded_files]
    df_raw = pd.concat(dfs, ignore_index=True)
    df_numeric = df_raw.select_dtypes(include=[np.number]).dropna()
    
    # --- VÁLVULA DE SEGURIDAD (ANTI-CRASH) ---
    MAX_ROWS = 5000
    if len(df_numeric) > MAX_ROWS:
        st.warning(f"⚠️ Dataset gigante detectado ({len(df_numeric)} filas). Para garantizar la fluidez y no saturar el servidor, se ha extraído una muestra representativa de {MAX_ROWS} filas para el análisis matemático.")
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

    # --- RENDERIZADO EDA ---
    if st.session_state.show_timeseries and 'ts_cols' in st.session_state:
        if len(st.session_state.ts_cols) > 0:
            with st.expander("📈 Series de Tiempo", expanded=True):
                with st.spinner("Graficando..."):
                    fig, axes = plt.subplots(len(st.session_state.ts_cols), 1, figsize=(10, 3 * len(st.session_state.ts_cols)))
                    if len(st.session_state.ts_cols) == 1: axes = [axes]
                    for ax, col in zip(axes, st.session_state.ts_cols):
                        ax.scatter(df.index, df[col], s=2, color='black')
                        ax.set_ylabel(col)
                    axes[-1].set_xlabel('Index / Sample')
                    plt.tight_layout()
                    st.pyplot(fig)

    if st.session_state.show_pairplot:
        with st.expander("📊 Pairplot con Densidad de Kernel (KDE)", expanded=True):
            with st.spinner("Generando Pairplot..."):
                cols_to_plot = df_numeric.columns[:10]
                fig = sns.pairplot(df[cols_to_plot], diag_kind="kde", markers=".", height=1.5)
                st.pyplot(fig)

    if st.session_state.show_correlation:
        with st.expander("🔥 Mapa de Correlación Phi_K", expanded=True):
            with st.spinner("Calculando dependencias no lineales..."):
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
                st.success("Características extraídas con éxito.")
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

st.markdown("---")
# ==========================================
# 4. LABORATORIO INTERACTIVO (INDEPENDIENTE)
# ==========================================
st.header("4. Laboratorio Interactivo: Complejidad del Modelo")
st.markdown("Este módulo funciona independientemente de los datos cargados. Mueve el control deslizante para cambiar el grado del polinomio. Observa cómo un modelo muy simple ignora la tendencia (Subajuste) y un modelo excesivamente complejo intenta capturar todo el ruido (Sobreajuste).")

np.random.seed(42)
X_sim = np.sort(np.random.rand(50, 1) * 10, axis=0)
y_sim = np.sin(X_sim).ravel() + np.random.randn(50) * 0.4

grado = st.slider("⚙️ Grado del Polinomio (Ajusta la barra)", min_value=1, max_value=15, value=1)

poly_features = PolynomialFeatures(degree=grado, include_bias=False)
X_poly = poly_features.fit_transform(X_sim)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_sim)

X_new = np.linspace(0, 10, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

fig_lab, ax_lab = plt.subplots(figsize=(10, 4))
sns.scatterplot(x=X_sim.ravel(), y=y_sim, color="black", label="Datos Crudos (Ruido)", s=40, ax=ax_lab)
ax_lab.plot(X_new, y_new, color="red", linewidth=2, label=f"Modelo Predictivo (Grado {grado})")
ax_lab.set_ylim(-2, 2)
ax_lab.set_title(f"Impacto del Grado Polinómico (Complejidad: {grado})")
ax_lab.legend()
st.pyplot(fig_lab)
