import streamlit as st
from pathlib import Path

st.set_page_config(layout="wide", page_title="Dashboard EDA — Seguro de Salud")
st.title("Dashboard EDA — Seguro de Salud")

# Ruta a la carpeta de imágenes
img_dir = Path(__file__).parent / "data"

# Lista de imágenes y títulos
imagenes = [
    ("eda_01_perfil_demografico.png", "Perfil demográfico de los afiliados"),
    ("eda_02_tasa_utilizacion.png", "Tasa de utilización del seguro"),
    ("eda_03_distribucion_costos.png", "Distribución de costos"),
    ("eda_04_boxplots_costo_perfil.png", "Boxplots: Costo por perfil"),
    ("eda_05_condiciones_preexistentes.png", "Condiciones preexistentes"),
    ("eda_06_tipos_reclamacion.png", "Tipos de reclamación"),
    ("eda_07_geografico.png", "Análisis geográfico"),
    ("eda_08_correlaciones.png", "Mapa de correlaciones"),
    ("eda_09_heatmaps_cruzados.png", "Heatmaps cruzados"),
]

# Muestra las imágenes en el dashboard
for archivo, titulo in imagenes:
    st.header(titulo)
    st.image(str(img_dir / archivo), use_column_width=True)
    st.markdown("---")