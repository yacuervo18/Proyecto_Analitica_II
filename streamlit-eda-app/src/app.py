from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("/Users/yedisoncuervo/Desktop/PROYECTO_ANALITICA_II/modelo_final/glm_optimizado.pkl")
DIAGRAM_PATH_PREPROC = Path("/Users/yedisoncuervo/Desktop/PROYECTO_ANALITICA_II/Diagrama_Analitica2.drawio.svg")
DIAGRAM_PATH_MODELO = Path("/Users/yedisoncuervo/Desktop/PROYECTO_ANALITICA_II/Procesos_modelos.drawio.svg")

st.set_page_config(
    layout="wide",
    page_title="Simulador de Tarifa | SURA",
    page_icon="🩺",
)

st.markdown(
    """
<style>
html, body, .stApp {
    background: #ffffff;
    color: #0f172a;
}

.stApp {
    background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
}

header, #MainMenu, footer {
    visibility: hidden;
}

.page-shell {
    max-width: 1500px;
    margin: 0 auto;
}

.hero-logo {
    text-align: center;
    margin: 0.35rem 0 0.6rem;
}

.hero-logo img {
    height: 72px;
    object-fit: contain;
}

.hero-title {
    text-align: center;
    font-size: 2.1rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0.25rem 0 0.25rem;
    letter-spacing: -0.03em;
}

.hero-subtitle {
    text-align: center;
    color: #475569;
    font-size: 0.98rem;
    margin-bottom: 1.4rem;
}

.glass-card {
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    box-shadow: 0 12px 34px rgba(15, 23, 42, 0.06);
    padding: 1.15rem 1.15rem 1.05rem;
}

.section-label {
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.55rem;
    font-weight: 700;
}

.section-title {
    font-size: 1.08rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.25rem;
}

.section-note {
    color: #64748b;
    font-size: 0.9rem;
    margin-bottom: 0.85rem;
    line-height: 1.55;
}

.result-card {
    position: sticky;
    top: 1rem;
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #dbe4f0;
    border-radius: 24px;
    box-shadow: 0 18px 34px rgba(15, 23, 42, 0.08);
    padding: 1.15rem;
}

.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.85rem;
}

.result-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.42rem 0.7rem;
    border-radius: 999px;
    background: #ecfdf5;
    color: #047857;
    font-size: 0.78rem;
    font-weight: 700;
}

.metric-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 0.9rem 0.95rem;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
}

.metric-label {
    color: #64748b;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.metric-value {
    color: #0f172a;
    font-size: 1.45rem;
    line-height: 1.15;
    font-weight: 800;
}

.metric-hint {
    color: #64748b;
    font-size: 0.82rem;
    margin-top: 0.35rem;
    line-height: 1.45;
}

.feature-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1rem;
    height: 100%;
}

.feature-card h4 {
    margin: 0 0 0.7rem;
    font-size: 1rem;
    color: #0f172a;
}

.feature-card .small-help {
    color: #64748b;
    font-size: 0.84rem;
    line-height: 1.45;
}

.divider-soft {
    height: 1px;
    background: linear-gradient(90deg, rgba(148, 163, 184, 0.06), rgba(148, 163, 184, 0.45), rgba(148, 163, 184, 0.06));
    margin: 0.95rem 0;
}

/* Estilos generales de botones */
.stButton > button {
    border-radius: 14px;
    font-weight: 700;
    padding: 0.55rem 0.85rem;
    font-size: 0.92rem;
    background: linear-gradient(135deg, #005b96, #00a3e0);
    border: 1px solid #005b96;
    color: #ffffff;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #005b96, #00a3e0);
    border: 1px solid #005b96;
    color: #ffffff;
}

.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, #0b74b8, #24b3ea);
    border: 1px solid #0b74b8;
    color: #ffffff;
}

/* Botones de navegación más pequeños */
.nav-bar .stButton > button {
    padding: 0.3rem 0.4rem !important;
    font-size: 0.8rem !important;
    border-radius: 8px !important;
}

.nav-bar .stButton > button[kind="primary"],
.nav-bar .stButton > button[kind="secondary"] {
    border-width: 1px;
}

.nav-bar .nav-left .stButton > button {
    border-top-left-radius: 14px !important;
    border-bottom-left-radius: 14px !important;
}

.nav-bar .nav-right .stButton > button {
    border-top-right-radius: 14px !important;
    border-bottom-right-radius: 14px !important;
}

.nav-bar .nav-center .stButton > button {
    border-radius: 0 !important;
}

.input-caption {
    color: #64748b;
    font-size: 0.8rem;
    margin-top: -0.3rem;
    margin-bottom: 0.5rem;
}

/* Inputs y selectores en gris suave */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="select"] input {
    background-color: #e9eef5 !important;
    border-color: #b8c3d1 !important;
    color: #0f172a !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    color: #0f172a !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] {
    color: #0f172a !important;
}

label[data-testid="stWidgetLabel"] p {
    color: #334155 !important;
    font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model() -> object:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def money(value: float) -> str:
    formatted = f"{float(value):,.2f}"
    return "$" + formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def pick_feature_name(expected_features: list[str], *candidates: str) -> str:
    for candidate in candidates:
        if candidate in expected_features:
            return candidate
    return candidates[0]


def build_feature_row(expected_features: list[str], inputs: dict) -> pd.DataFrame:
    row = {feature: 0 for feature in expected_features}

    feature_map = {
        "edad": pick_feature_name(expected_features, "edad"),
        "sexo_m": pick_feature_name(expected_features, "Sexo_Cd_limpio_M"),
        "sexo_nb": pick_feature_name(expected_features, "Sexo_Cd_limpio_NOBINARIO"),
        "cancer": pick_feature_name(expected_features, "CANCER"),
        "diabetes": pick_feature_name(expected_features, "DIABETES"),
        "cardiaca": pick_feature_name(expected_features, "ENF_CARDIACA"),
        "hipertension": pick_feature_name(expected_features, "HIPERTENSION"),
        "pulmonar": pick_feature_name(expected_features, "ENF_PULMONAR"),
        "num_condiciones": pick_feature_name(expected_features, "num_condiciones"),
        "cali": pick_feature_name(expected_features, "CIUDAD_NORM_CALI"),
        "cartagena": pick_feature_name(expected_features, "CIUDAD_NORM_CARTAGENA"),
        "medellin": pick_feature_name(expected_features, "CIUDAD_NORM_MEDELLIN"),
        "sin_info": pick_feature_name(
            expected_features,
            "CIUDAD_NORM_SIN_INFORMACION",
            "CIUDAD_NORM_SIN INFORMACION",
        ),
        "meses": pick_feature_name(expected_features, "meses_expuesto_total"),
    }

    row[feature_map["edad"]] = int(inputs["edad"])
    row[feature_map["sexo_m"]] = int(inputs["sexo_m"])
    row[feature_map["sexo_nb"]] = int(inputs["sexo_nb"])
    row[feature_map["cancer"]] = int(inputs["cancer"])
    row[feature_map["diabetes"]] = int(inputs["diabetes"])
    row[feature_map["cardiaca"]] = int(inputs["cardiaca"])
    row[feature_map["hipertension"]] = int(inputs["hipertension"])
    row[feature_map["pulmonar"]] = int(inputs["pulmonar"])
    row[feature_map["num_condiciones"]] = int(inputs["num_condiciones"])
    row[feature_map["cali"]] = int(inputs["cali"])
    row[feature_map["cartagena"]] = int(inputs["cartagena"])
    row[feature_map["medellin"]] = int(inputs["medellin"])
    row[feature_map["sin_info"]] = int(inputs["sin_info"])
    row[feature_map["meses"]] = int(inputs["meses_expuesto_total"])

    return pd.DataFrame([row], columns=expected_features)


def predict_pure_premium(model: object, input_df: pd.DataFrame) -> float:
    prediction_log = float(model.predict(input_df)[0])
    prediction = float(np.expm1(prediction_log))
    return max(prediction, 0.0)


def initialize_state() -> None:
    defaults = {
        "prima_pura_anual": None,
        "prima_comercial_anual": None,
        "prima_comercial_mensual": None,
        "input_row": None,
        "cliente_nombre": "",
        "cliente_documento": "",
        "margen_comercial": 0.25,
        "last_margin_pct": None,
        "active_page": "simulacion",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_metric(label: str, value: str, hint: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-hint">{hint}</div>' if hint else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


initialize_state()

try:
    model = load_model()
    expected_features = list(
        getattr(
            model,
            "feature_names_in_",
            [
                "edad",
                "Sexo_Cd_limpio_M",
                "Sexo_Cd_limpio_NOBINARIO",
                "CANCER",
                "DIABETES",
                "ENF_CARDIACA",
                "HIPERTENSION",
                "ENF_PULMONAR",
                "num_condiciones",
                "CIUDAD_NORM_CALI",
                "CIUDAD_NORM_CARTAGENA",
                "CIUDAD_NORM_MEDELLIN",
                "CIUDAD_NORM_SIN_INFORMACION",
                "meses_expuesto_total",
            ],
        )
    )
except Exception as error:
    st.error(f"No se pudo cargar el modelo final: {error}")
    st.stop()

st.markdown(
    """
    <div class="page-shell">
        <div class="hero-logo">
            <img src="https://upload.wikimedia.org/wikipedia/commons/6/61/Seguros_SURA_Logo.svg" alt="SURA" />
        </div>
        <div class="hero-title">Simulador de Tarifa</div>
        <div class="hero-subtitle">Cotizador de prima pura y prima comercial para seguro de salud.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ======================= BARRA DE NAVEGACIÓN =======================
st.markdown('<div class="nav-bar">', unsafe_allow_html=True)
nav_col1, nav_col2, nav_col3, _ = st.columns([0.8, 0.8, 0.8, 4], gap="small")

with nav_col1:
    st.markdown('<div class="nav-left">', unsafe_allow_html=True)
    if st.button(
        "Proceso",
        use_container_width=True,
        type="primary" if st.session_state.active_page == "preprocesamiento" else "secondary",
    ):
        st.session_state.active_page = "preprocesamiento"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with nav_col2:
    st.markdown('<div class="nav-center">', unsafe_allow_html=True)
    if st.button(
        "Modelos",
        use_container_width=True,
        type="primary" if st.session_state.active_page == "modelo" else "secondary",
    ):
        st.session_state.active_page = "modelo"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with nav_col3:
    st.markdown('<div class="nav-right">', unsafe_allow_html=True)
    if st.button(
        "Simulador",
        use_container_width=True,
        type="primary" if st.session_state.active_page == "simulacion" else "secondary",
    ):
        st.session_state.active_page = "simulacion"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ======================= PÁGINA PREPROCESAMIENTO =======================
if st.session_state.active_page == "preprocesamiento":
    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Preprocesamiento</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ETL y transformación de datos</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Diagrama del flujo de extracción, limpieza y preparación de las variables utilizadas por el modelo.</div>',
        unsafe_allow_html=True,
    )
    if DIAGRAM_PATH_PREPROC.exists():
        st.image(str(DIAGRAM_PATH_PREPROC), use_container_width=True)
    else:
        st.warning(f"No se encontró la imagen en: {DIAGRAM_PATH_PREPROC}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ======================= PÁGINA MODELO =======================
if st.session_state.active_page == "modelo":
    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Modelo Estadístico</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Modelo Elegido - GLM optimizado para prima pura</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Modelado, análisis de resultados, selección del mejor modelo y validación.</div>',
        unsafe_allow_html=True,
    )
    if DIAGRAM_PATH_MODELO.exists():
        st.image(str(DIAGRAM_PATH_MODELO), use_container_width=True)
    else:
        st.warning(f"No se encontró la imagen en: {DIAGRAM_PATH_MODELO}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ========== NUEVA SECCIÓN: VALIDACIÓN MONTE CARLO ==========
    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Validación del Modelo</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Simulación Monte Carlo del GLM Gamma</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Se realizaron 200 iteraciones de entrenamiento/validación para evaluar la estabilidad del modelo. Los resultados se muestran a continuación.</div>',
        unsafe_allow_html=True,
    )

    # Tabla de resultados
    montecarlo_stats = {
        "Métrica": ["R² medio", "Desviación estándar", "R² mínimo", "R² máximo", "Percentil 2.5", "Percentil 97.5"],
        "Valor": [0.856173, 0.004401, 0.838003, 0.867515, 0.846483, 0.864207]
    }
    df_stats = pd.DataFrame(montecarlo_stats)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

    # Texto de robustez
    st.markdown(
        """
        <div style="background-color: #f0f9f0; padding: 1rem; border-radius: 14px; margin-top: 0.8rem;">
            <p style="font-weight: 700; margin-bottom: 0.5rem; color: #0b5e2e;">Conclusión sobre la robustez del modelo</p>
            <p style="margin-bottom: 0; color: #1e4620;">
            El modelo GLM Gamma presenta una <strong>desviación estándar muy baja (0.0044)</strong> en el R², lo que indica una 
            <strong>alta estabilidad</strong> ante diferentes particiones de los datos. El intervalo de confianza del 95% para el R² es 
            [0.8465 – 0.8642], estrecho y alejado de valores bajos. Por tanto, el modelo es <strong>robusto</strong>, generaliza correctamente 
            y puede desplegarse con confianza en entornos productivos.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ======================= SIMULADOR =======================
main_col, result_col = st.columns([2.2, 1], gap="large")

with main_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">1. Identificación</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Datos basicos del asegurado</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Campos informativos para trazabilidad. Estos datos no se envían al modelo.</div>', unsafe_allow_html=True)

    id_col1, id_col2 = st.columns(2)
    with id_col1:
        nombres = st.text_input("Nombres", placeholder="Ej. Ana María")
    with id_col2:
        apellidos = st.text_input("Apellidos", placeholder="Ej. Pérez Gómez")

    id_col3, id_col4 = st.columns(2)
    with id_col3:
        tipo_documento = st.selectbox("Tipo de documento", ["CC", "TI", "CE", "Pasaporte", "Otro"], index=0)
    with id_col4:
        numero_documento = st.text_input("Número de documento", placeholder="Ej. 123456789")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">2. Variables requeridas por el modelo</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Perfil técnico para cotización</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Estas son las variables técnicas usadas por el motor de tarifación.</div>', unsafe_allow_html=True)

    dem_col1, dem_col2, dem_col3, dem_col4 = st.columns([1.2, 1, 1, 1])
    with dem_col1:
        st.markdown("**EDAD**")
        edad = st.number_input("EDAD", min_value=0, max_value=120, value=35, step=1, label_visibility="collapsed")
    with dem_col2:
        st.markdown("**SEXO**")
        sexo = st.selectbox("SEXO", ["Femenino", "Masculino", "Desconocido"], index=0, label_visibility="collapsed")
    with dem_col3:
        st.markdown("**CIUDAD**")
        ciudad = st.selectbox(
            "CIUDAD",
            [
                "Bogotá",
                "Medellín",
                "Cali",
                "Cartagena",
                "Barranquilla",
                "Bucaramanga",
                "Pereira",
                "Sin información",
                "Otra",
            ],
            index=0,
            label_visibility="collapsed",
        )
    with dem_col4:
        st.markdown("**MESES EXPUESTO TOTAL**")
        meses_expuesto_total = st.number_input(
            "MESES EXPUESTO TOTAL",
            min_value=1,
            max_value=12,
            value=12,
            step=1,
            label_visibility="collapsed",
        )

    st.markdown("**CONDICIONES PREEXISTENTES**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("**CANCER**")
        cancer = st.checkbox("CANCER", value=False, key="cond_cancer", label_visibility="collapsed")
    with c2:
        st.markdown("**DIABETES**")
        diabetes = st.checkbox("DIABETES", value=False, key="cond_diabetes", label_visibility="collapsed")
    with c3:
        st.markdown("**ENF_CARDIACA**")
        enf_cardiaca = st.checkbox("ENF_CARDIACA", value=False, key="cond_cardiaca", label_visibility="collapsed")
    with c4:
        st.markdown("**HIPERTENSION**")
        hipertension = st.checkbox("HIPERTENSION", value=False, key="cond_hipertension", label_visibility="collapsed")
    with c5:
        st.markdown("**ENF_PULMONAR**")
        enf_pulmonar = st.checkbox("ENF_PULMONAR", value=False, key="cond_pulmonar", label_visibility="collapsed")

    num_condiciones = int(sum([cancer, diabetes, enf_cardiaca, hipertension, enf_pulmonar]))
    st.markdown('<div class="input-caption">num_condiciones se calcula automáticamente como la suma de las condiciones clínicas seleccionadas.</div>', unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">num_condiciones</div>
                <div class="metric-value">{num_condiciones}</div>
                <div class="metric-hint">Variable derivada para el GLM</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cc2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Sexo codificado</div>
                <div class="metric-value">{sexo}</div>
                <div class="metric-hint">Femenino = 0/0, Masculino = 1/0, No binario = 0/1</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cc3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Ciudad seleccionada</div>
                <div class="metric-value">{ciudad}</div>
                <div class="metric-hint">Se convierte internamente en dummies</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)
    compute_pure = st.button("Calcular prima pura", type="secondary", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    if compute_pure:
        sexo_m = 1 if sexo == "Masculino" else 0
        sexo_nb = 1 if sexo == "No binario" else 0
        city_map = {
            "Bogotá": None,
            "Cali": "cali",
            "Cartagena": "cartagena",
            "Medellín": "medellin",
            "Sin información": "sin_info",
            "Barranquilla": None,
            "Bucaramanga": None,
            "Pereira": None,
            "Otra": None,
        }
        city_flags = {"cali": 0, "cartagena": 0, "medellin": 0, "sin_info": 0}
        selected_city_flag = city_map[ciudad]
        if selected_city_flag is not None:
            city_flags[selected_city_flag] = 1

        feature_row = build_feature_row(
            expected_features,
            {
                "edad": int(edad),
                "sexo_m": sexo_m,
                "sexo_nb": sexo_nb,
                "cancer": int(cancer),
                "diabetes": int(diabetes),
                "cardiaca": int(enf_cardiaca),
                "hipertension": int(hipertension),
                "pulmonar": int(enf_pulmonar),
                "num_condiciones": num_condiciones,
                "cali": city_flags["cali"],
                "cartagena": city_flags["cartagena"],
                "medellin": city_flags["medellin"],
                "sin_info": city_flags["sin_info"],
                "meses_expuesto_total": int(meses_expuesto_total),
            },
        )

        prima_pura_anual = predict_pure_premium(model, feature_row)

        st.session_state.prima_pura_anual = prima_pura_anual
        st.session_state.input_row = feature_row
        st.session_state.cliente_nombre = f"{nombres} {apellidos}".strip()
        st.session_state.cliente_documento = f"{tipo_documento} {numero_documento}".strip()
        st.session_state.prima_comercial_anual = None
        st.session_state.prima_comercial_mensual = None

        st.toast("Prima pura calculada correctamente", icon="✅")

with result_col:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-header">
            <div>
                <div class="section-label" style="margin-bottom:0.25rem;">3. Resultado</div>
                <div class="section-title" style="margin-bottom:0.1rem;">Resumen de cotización</div>
                <div class="section-note" style="margin-bottom:0;">La prima pura se mantiene visible aquí, abajo a la derecha.</div>
            </div>
            <div class="result-chip">Motor de tarifación activo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown(f"<h4>Cliente</h4>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="small-help">
            <strong>{st.session_state.cliente_nombre or 'Sin nombre aún'}</strong><br>
            {st.session_state.cliente_documento or 'Documento no diligenciado'}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)

    if st.session_state.prima_pura_anual is not None:
        render_metric(
            "Prima pura anual",
            money(st.session_state.prima_pura_anual),
            "Estimación anual del costo esperado del asegurado.",
        )
        st.markdown('<div style="height:0.7rem"></div>', unsafe_allow_html=True)
        render_metric(
            "Prima pura mensual",
            money(st.session_state.prima_pura_anual / 12),
            "Referencia mensual derivada de la prima pura anual.",
        )
    else:
        st.markdown(
            """
            <div class="feature-card">
                <h4>Prima pura</h4>
                <div class="small-help">Completa la información y presiona <strong>Calcular prima pura</strong> para ver la estimación.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:0.95rem"></div>', unsafe_allow_html=True)

    loading_pct = st.slider(
        "Carga comercial total (%) -> gastos + utilidad",
        min_value=0.0,
        max_value=90.0,
        value=float(st.session_state.margen_comercial * 100),
        step=1.0,
        help="Suma de gastos + utilidad sobre la tarifa final (p = g + u).",
    )
    st.session_state.margen_comercial = loading_pct / 100.0

    if st.session_state.last_margin_pct != loading_pct:
        st.session_state.prima_comercial_anual = None
        st.session_state.prima_comercial_mensual = None
        st.session_state.last_margin_pct = loading_pct

    quote_commercial = st.button("Cotizar prima comercial", type="primary", use_container_width=True)

    if quote_commercial:
        if st.session_state.prima_pura_anual is None:
            st.warning("Primero calcula la prima pura para poder cotizar la prima comercial.")
        elif st.session_state.margen_comercial >= 1.0:
            st.warning("La carga comercial debe ser menor al 100% para poder calcular la tarifa.")
        else:
            prima_comercial_anual = st.session_state.prima_pura_anual / (1 - st.session_state.margen_comercial)
            prima_comercial_mensual = prima_comercial_anual / 12

            st.session_state.prima_comercial_anual = prima_comercial_anual
            st.session_state.prima_comercial_mensual = prima_comercial_mensual

    if st.session_state.prima_comercial_anual is not None:
        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
        render_metric(
            "Prima comercial anual",
            money(st.session_state.prima_comercial_anual),
            "Calculada como S / (1 - p), donde p = gastos + utilidad.",
        )
        st.markdown('<div style="height:0.7rem"></div>', unsafe_allow_html=True)
        render_metric(
            "Prima comercial mensual",
            money(st.session_state.prima_comercial_mensual),
            "Valor final de pago mensual para el cliente.",
        )
    elif st.session_state.prima_pura_anual is not None:
        st.markdown(
            """
            <div class="feature-card" style="margin-top:0.9rem;">
                <h4>Prima comercial</h4>
                <div class="small-help">Presiona <strong>Cotizar prima comercial</strong> para convertir la prima pura en el valor comercial final.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="feature-card" style="margin-top:0.9rem;">
                <h4>Prima comercial</h4>
                <div class="small-help">Se calcula después de obtener la prima pura.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)