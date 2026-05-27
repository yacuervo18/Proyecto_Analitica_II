import streamlit as st
st.set_page_config(
    layout="wide",
    page_title="UdeA Insurance — Avance 1",
)

# ── Resto de imports después de set_page_config ────────────────
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# CSS GLOBAL
st.markdown("""
<style>
.metric-card {
    background: white; border-radius: 14px; padding: 1.2rem 0.8rem;
    border: 1px solid #E2E8F0; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    height: 105px; display: flex; flex-direction: column; justify-content: center;
}
.metric-val { font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
.metric-lab { font-size: 0.74rem; color: #64748B; margin-top: 5px; line-height: 1.4; }

.card        { background: white; border-radius: 14px; padding: 1.4rem;
               border: 1px solid #E2E8F0; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.card-teal   { border-left: 5px solid #0D9488; }
.card-orange { border-left: 5px solid #F97316; }
.card-purple { border-left: 5px solid #7C3AED; }
.card-blue   { border-left: 5px solid #0891B2; }
.card-red    { border-left: 5px solid #EF4444; background: #FEF2F2; }
.card-green  { border-left: 5px solid #22C55E; background: #F0FDF4; }
.card-yellow { border-left: 5px solid #F59E0B; background: #FFFBEB; }

.formula {
    background: #0B1F3A; color: white;
    padding: 1rem 1.5rem; border-radius: 12px;
    font-family: 'Courier New', monospace; font-size: 1rem;
    text-align: center; border: 1px solid #0D9488; margin: 0.8rem 0;
}
.pipe-done    { background:#F0FDFA; border:2px solid #0D9488; color:#0D9488;
                border-radius:10px; padding:0.6rem 0.3rem; text-align:center;
                font-size:0.78rem; font-weight:600; }
.pipe-current { background:#FEF3C7; border:2px solid #F59E0B; color:#92400E;
                border-radius:10px; padding:0.6rem 0.3rem; text-align:center;
                font-size:0.78rem; font-weight:600; }
.pipe-future  { background:#F1F5F9; border:2px solid #CBD5E1; color:#94A3B8;
                border-radius:10px; padding:0.6rem 0.3rem; text-align:center;
                font-size:0.78rem; font-weight:600; }
.insight { background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px;
           padding:0.9rem 1.2rem; margin-top:0.6rem; font-size:0.88rem;
           line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ESTADO DE SESIÓN
if "sec" not in st.session_state:
    st.session_state.sec = 0

# ENCABEZADO
st.markdown("""
<h1 style='color:white; font-size:2rem; font-weight:700; margin: 0 0 1.2rem 0; text-align:center;'>
  Reto de Tarifación Seguro de Salud - Avance 1
</h1>
""", unsafe_allow_html=True)

# BARRA DE NAVEGACIÓN
NAV = [
    ("1-", "Simulador\nde Tarifa"),
]
nav_cols = st.columns(1)
for i, (col, (num, label)) in enumerate(zip(nav_cols, NAV)):
    with col:
        tipo = "primary" if st.session_state.sec == i else "secondary"
        if st.button(f"{num}\n{label}", key=f"nav{i}",
                     use_container_width=True, type=tipo):
            st.session_state.sec = i
            st.rerun()
st.divider()

# SECCIÓN 1 — SIMULADOR DE TARIFA
if st.session_state.sec == 0:

    st.markdown("## Simulador de Tarifa")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="card card-orange">
          <h3 style="color:#F97316;margin:0 0 0.7rem">Problema de Negocio</h3>
          <p style="font-size:1rem;line-height:1.8;color:#1E293B">
            <strong>UdeA Insurance</strong> a través del equipo de analítica LIMBICO</strong> necesita saber cuánto cobrarle a cada persona
            por su seguro de salud, y hoy
            <strong style="color:#DC2626">no tiene un modelo para calcularlo con datos</strong>.
          </p>
          <hr style="border:none;border-top:1px solid #E2E8F0;margin:0.8rem 0">
          <p style="font-size:0.88rem;color:#475569;line-height:1.8">
            Hoy cobran una <strong>prima comercial igual para todos</strong>: una niña de 10 años sana
            paga lo mismo que un hombre de 70 con cáncer e hipertensión.
            Eso sería de cierta forma inutos e  insostenible financieramente.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card card-red" style="margin-top:1rem">
          <h4 style="color:#DC2626;margin:0 0 0.6rem">Consecuencias del problema</h4>
          <ul style="margin:0;padding-left:1.2rem;color:#7F1D1D;line-height:2.2;font-size:0.9rem">
            <li>Si se cobra muy barato - la aseguradora pierde</li>
            <li>Si se cobra muy costoso a personas sanas - pierden clientes</li>
            <li>Sin modelo es muy probable un riesgo de <strong>insolvencia o pérdida de mercado</strong></li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card card-teal">
          <h3 style="color:#0D9488;margin:0 0 0.7rem">Problema Analítico</h3>
          <p style="font-size:1rem;line-height:1.8;color:#1E293B">
            Construir un <strong>modelo predictivo</strong> que estime el costo esperado
            de siniestros por asegurado en función de su perfil demográfico,
            condiciones preexistentes y tiempo de exposición.
          </p>
          <hr style="border:none;border-top:1px solid #E2E8F0;margin:0.8rem 0">
          <p style="font-size:0.88rem;color:#475569;line-height:1.8">
            Posibles variables predictoras: <code>edad</code> · <code>sexo</code> ·
            <code>ciudad</code> · <code>condiciones preexistentes</code> ·
            <code>meses expuesto</code>
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card card-green" style="margin-top:1rem">
          <h4 style="color:#166534;margin:0 0 0.5rem">La conexión  del negocio  con analítica</h4>
          <p style="color:#166534;margin:0;line-height:1.8;font-size:0.9rem">
            Si predecimos el <strong>costo esperado</strong> de cada perfil,
            calculamos la <strong>prima justa</strong> con la fórmula actuarial:
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula">
        <span style="color:#14B8A6;font-weight:700">Tarifa</span>
        <span style="color:#94A3B8"> = </span>
        <span style="color:#FCD34D;font-weight:700">Prima Pura</span>
        <span style="color:#94A3B8"> + Gastos + Utilidad</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span style="color:#FCD34D;font-weight:700">Prima Pura</span>
        <span style="color:#94A3B8"> = </span>
        <span style="color:#F97316;font-weight:700">Frecuencia</span>
        <span style="color:#94A3B8"> × </span>
        <span style="color:#A78BFA;font-weight:700">Severidad</span>
        <span style="color:#94A3B8"> </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Calculo de la prima pura (El mismo valor para todos) con los datos reales:")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (val, lab, color) in zip([c1, c2, c3, c4, c5], [
        ("219.800", "reclamantes", "#0D9488"),
        ("÷ 260.853", "asegurados totales", "#64748B"),
        ("= 0.843", "Frecuencia real (84.3%)", "#F97316"),
        ("× $5.434.389", "Severidad promedio", "#7C3AED"),
        ("$4.581.170/año", "Prima pura calculada", "#0891B2"),
    ]):
        with col:
            sz = "1rem" if len(val) > 10 else "1.3rem"
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-val" style="color:{color};font-size:{sz}">{val}</div>
              <div class="metric-lab">{lab}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight" style="color:#1E293B">
      <strong></strong> Hoy UdeA Insurance cobra
      <strong>$381.739/mes igual para todos</strong>.
      El modelo va a personalizar ese valor: una persona sana de 20 años pagará mucho menos;
      alguien con cáncer e hipertensión pagará acorde a su riesgo real.
    </div>
    """, unsafe_allow_html=True)
