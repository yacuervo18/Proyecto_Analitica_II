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
    ("1-", "Problema\nde Negocio"),
    ("2-", "Diseño de\nla Solución"),
    ("3-", "Limpieza y\nUnificación"),
    ("4-", "Hallazgos\ndel EDA"),
    ("5-", "Próximos\nPasos"),
]
nav_cols = st.columns(5)
for i, (col, (num, label)) in enumerate(zip(nav_cols, NAV)):
    with col:
        tipo = "primary" if st.session_state.sec == i else "secondary"
        if st.button(f"{num}\n{label}", key=f"nav{i}",
                     use_container_width=True, type=tipo):
            st.session_state.sec = i
            st.rerun()
st.divider()

# SECCIÓN 1 — PROBLEMA DE NEGOCIO Y ANALÍTICO
if st.session_state.sec == 0:

    st.markdown("## Problema de Negocio y Problema Analítico")

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


# SECCIÓN 2 — DISEÑO DE LA SOLUCIÓN
elif st.session_state.sec == 1:

    st.markdown("## Diseño de la Solución — Mapa del Proceso y Limpieza de Datos")

    # Pipeline visual
    etapas = [
        ("BDs\nCrudas",      "320K+261K\n+1.9M filas",  "done"),
        ("Limpieza\nIndiv.", "Fechas·Nulos\nInconsist.", "done"),
        ("Unión\n3 BDs",     "1 tabla\n260.853 filas",   "done"),
        ("Tabla\nMaestra",   "32 variables\nlimpias",    "done"),
        ("EDA",              "5 preguntas\nnegocio",     "done"),
        ("Selección\nVars.", "32→11\nfeatures",          "current"),
        ("Modelado",         "Frec×Sev\nML",             "future"),
        ("Ayuda\nVentas",    "Streamlit\ninteractivo",   "future"),
    ]

    cols_pipe = st.columns(len(etapas))
    for i, (col, (label, sub, estado)) in enumerate(zip(cols_pipe, etapas)):
        with col:
            icono = "✅" if estado == "done" else ("⚡" if estado == "current" else "○")
            st.markdown(f"""
            <div class="pipe-{estado}">
              <div style="font-size:1.3rem">{icono}</div>
              <div style="white-space:pre-line;margin-top:4px">{label}</div>
              <div style="font-size:0.65rem;color:#64748B;margin-top:3px;white-space:pre-line">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;gap:2rem;margin:0.8rem 0 1.2rem;font-size:0.83rem">
      <span style="color:#0D9488;font-weight:600">✅ Completado — Avance 1</span>
      <span style="color:#F59E0B;font-weight:600">⚡ En progreso — ESTAMOS AQUÍ</span>
      <span style="color:#94A3B8;font-weight:600">○ Pendiente — Avances 2 y 3</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Las 3 fuentes de datos y qué se hizo en cada una")

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown("""
        <div class="card card-teal">
          <h4 style="color:#0D9488;margin:0 0 0.6rem"> BD_Exposicion</h4>
          <div style="display:flex;gap:6px;margin-bottom:0.7rem;flex-wrap:wrap">
            <span style="background:#F0FDFA;color:#0D9488;padding:2px 8px;border-radius:20px;font-size:0.72rem;font-weight:600">320.002 filas</span>
            <span style="background:#F0FDFA;color:#0D9488;padding:2px 8px;border-radius:20px;font-size:0.72rem;font-weight:600">5 cols</span>
          </div>
          <p style="font-size:0.82rem;color:#475569;font-style:italic;margin:0 0 0.6rem">
            ¿Cuánto tiempo estuvo asegurada cada persona?
          </p>
          <ul style="font-size:0.8rem;color:#374151;margin:0;padding-left:1.1rem;line-height:2.1">
            <li>Conversión de fechas a datetime</li>
            <li>FECHA_SALIDA = cancelación → si no, FIN</li>
            <li>Corrección de 18 inconsistencias</li>
            <li>Cálculo <code>dias/meses_expuesto</code></li>
          </ul>
          <div style="background:#FFFBEB;border-radius:8px;padding:8px 10px;margin-top:0.7rem;font-size:0.78rem;color:#92400E">
            <strong>Hallazgo:</strong> 65.93% nulos en FECHA_CANCELACION
            - nunca cancelaron, NO es error
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card card-orange">
          <h4 style="color:#F97316;margin:0 0 0.6rem"> BD_SocioDemografica</h4>
          <div style="display:flex;gap:6px;margin-bottom:0.7rem;flex-wrap:wrap">
            <span style="background:#FFF7ED;color:#C2410C;padding:2px 8px;border-radius:20px;font-size:0.72rem;font-weight:600">261.267 filas</span>
            <span style="background:#FFF7ED;color:#C2410C;padding:2px 8px;border-radius:20px;font-size:0.72rem;font-weight:600">9 cols</span>
          </div>
          <p style="font-size:0.82rem;color:#475569;font-style:italic;margin:0 0 0.6rem">
            ¿Quién es el asegurado y qué condiciones tiene?
          </p>
          <ul style="font-size:0.8rem;color:#374151;margin:0;padding-left:1.1rem;line-height:2.1">
            <li>FechaNacimiento: <code>edad</code> en años</li>
            <li>CIUDAD normalizada (sin tildes)</li>
            <li>Sexo: F / M / NOBINARIO</li>
            <li><code>num_condiciones</code> = suma de 5</li>
          </ul>
          <div style="background:#FFFBEB;border-radius:8px;padding:8px 10px;margin-top:0.7rem;font-size:0.78rem;color:#92400E">
              <strong>Hallazgo:</strong> 36 registros no binarios
            - categoría NOBINARIO creada
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card card-purple">
          <h4 style="color:#7C3AED;margin:0 0 0.6rem"> BD_Siniestros</h4>
          <div style="display:flex;gap:6px;margin-bottom:0.7rem;flex-wrap:wrap">
            <span style="background:#F5F3FF;color:#5B21B6;padding:2px 8px;border-radius:20px;font-size:0.72rem;font-weight:600">1.919.949 filas</span>
            <span style="background:#F5F3FF;color:#5B21B6;padding:2px 8px;border-radius:20px;font-size:0.72rem;font-weight:600">7 cols</span>
          </div>
          <p style="font-size:0.82rem;color:#475569;font-style:italic;margin:0 0 0.6rem">
            ¿Qué reclamó, cuándo y cuánto costó?
          </p>
          <ul style="font-size:0.8rem;color:#374151;margin:0;padding-left:1.1rem;line-height:2.1">
            <li>Fecha_Reclamacion: datetime</li>
            <li>Valores negativos: clip(0)</li>
            <li>Dummies por tipo: <code>rec_*</code></li>
            <li>Agregado por afiliado</li>
          </ul>
          <div style="background:#FFFBEB;border-radius:8px;padding:8px 10px;margin-top:0.7rem;font-size:0.78rem;color:#92400E">
              <strong>Hallazgo:</strong> 1.9M filas - 240.108 afiliados
            únicos con ≥1 siniestro
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Estrategia de unión — LEFT JOIN doble")

    col_a, sp1, col_b, sp2, col_c = st.columns([2, 0.25, 2, 0.25, 2])
    for col, icon, title, sub, color in [
    (col_a, "SocioDemografica", "261.265 afiliados · tabla base", "Tabla base", "#F97316"),
    (col_b, "Exposicion",       "LEFT JOIN · Asegurado_Id = Afiliado_Id", "Unión exposición", "#0D9488"),
    (col_c, "Siniestros",       "LEFT JOIN · Afiliado_Id = Afiliado_Id", "Unión siniestros", "#7C3AED"),
    ]:
       with col:
          st.markdown(f"""
          <div class="card" style="text-align:center;border-top:4px solid {color}">
            <div style="font-size:1.6rem; font-weight:700; color:{color}">{icon}</div>
            <div style="color:#1E293B;margin-top:4px">{title}</div>
            <div style="font-size:0.78rem;color:#64748B;margin-top:4px;line-height:1.5">{sub}</div>
          </div>
          """, unsafe_allow_html=True)
  
    with sp1:
        st.markdown("<div style='text-align:center;font-size:2rem;padding-top:1.2rem'>⊕</div>", unsafe_allow_html=True)
    with sp2:
        st.markdown("<div style='text-align:center;font-size:2rem;padding-top:1.2rem'>⊕</div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="insight" style="color:#1E293B">
          <strong>¿Por qué LEFT JOIN?</strong> Para conservar a los 41.053 asegurados que
          nunca reclamaron. Su costo = $0 es información válida: son personas sanas.
          Un INNER JOIN los eliminaría y sesgaría el modelo.
        </div>
        """, unsafe_allow_html=True)

# SECCIÓN 3 — LIMPIEZA Y UNIFICACIÓN
elif st.session_state.sec == 2:

    st.markdown("## Limpieza y Unificación — Resultados Reales")

    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown("### Cobertura de datos entre bases de datos")

        categorias = [
            "Solo Socio.", "Solo Exp.", "Solo Sin.",
            "Socio+Exp.", "Socio+Sin.",
            "COMPLETO\n(Socio+Exp+Sin)"
        ]
        valores = [261265, 293288, 240108, 260853, 219812, 219800]
        colores = ["#94A3B8", "#0D9488", "#7C3AED", "#0891B2", "#F97316", "#22C55E"]

        fig = go.Figure(go.Bar(
            x=categorias, y=valores, marker_color=colores,
            text=[f"{v:,}" for v in valores], textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Categoría de cobertura",
            yaxis_title="Número de usuarios",
            plot_bgcolor="white", paper_bgcolor="white",
            height=350, margin=dict(t=20, b=10),
            yaxis=dict(gridcolor="#F1F5F9", tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
            xaxis=dict(tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
            font=dict(color="#1E293B")
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Detalle de intersecciones")
        df_cob = pd.DataFrame({
            "Descripción": [
                "Total en Sociodemográficos", "Total en Exposición", "Total en Siniestros",
                "Socio + Exposición", "Socio + Siniestros", "Socio + Exp. + Sin. (COMPLETO)",
                "Solo Sociodemog. (sin ninguna otra)",
                "Socio + Exp., SIN siniestros", "Socio + Sin., SIN exposición",
            ],
            "Usuarios": [
                "261.265", "293.288", "240.108",
                "260.853", "219.812", "219.800",
                "400", "41.053", "12",
            ],
        })
        st.dataframe(df_cob, hide_index=True, use_container_width=True, height=300)

    with col2:
        st.markdown("### Resumen Ejecutivo de la Cartera")

        metricas = [
            ("260.853",           "Total asegurados\nanalizados",       "#0D9488"),
            ("219.800  (84.3%)",  "Usaron el seguro\n≥1 reclamación",  "#22C55E"),
            ("41.053  (15.7%)",   "No usaron\nel seguro",               "#F97316"),
            ("$1.19 Billones",    "Costo total\npagado (COP)",          "#7C3AED"),
            ("$5.434.389",        "Costo promedio\nquienes usaron",     "#0891B2"),
            ("$1.512.496",        "Costo mediano\nquienes usaron",      "#0891B2"),
            ("8.3",               "Reclamaciones prom.\npor usuario",   "#F97316"),
            ("11.2 meses",        "Exposición prom.\nde la cartera",    "#0D9488"),
            ("35.6 años",         "Edad promedio\nde la cartera",       "#64748B"),
        ]

        for k in range(0, len(metricas), 3):
            fila = metricas[k:k+3]
            inner_cols = st.columns(len(fila))
            for icol, (val, lab, color) in zip(inner_cols, fila):
                with icol:
                    sz = "0.95rem" if len(val) > 12 else "1.3rem"
                    st.markdown(f"""
                    <div class="metric-card" style="height:90px">
                      <div class="metric-val" style="color:{color};font-size:{sz}">{val}</div>
                      <div class="metric-lab">{lab}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card card-yellow" style="margin-top:0.5rem;text-align:center">
          <div style="font-size:2.6rem;font-weight:700;color:#F59E0B">x3.6</div>
          <div style="font-size:0.88rem;color:#92400E;font-weight:600">Brecha Promedio / Mediana</div>
          <div style="font-size:0.78rem;color:#78350F;margin-top:6px;line-height:1.6">
            Distribución muy asimétrica (se considera normal en seguros de salud).<br>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("###  Percentiles del costo real (quienes reclamaron)")

    df_pct = pd.DataFrame({
        "Percentil": ["P25", "P50 (Mediana)", "P75", "P90", "P95", "P99"],
        "Costo (COP)": ["$486.463", "$1.512.496", "$4.410.960",
                        "$11.717.115", "$20.239.760", "$56.674.465"],
        "Interpretación": [
            "El 25% más económico",
            "La mitad de los asegurados",
            "El 25% más costoso empieza aquí",
            "Solo el 10% supera este costo",
            "Solo el 5% supera este costo",
            "El 1% más costoso — altísimo impacto",
        ],
    })
    st.dataframe(df_pct, hide_index=True, use_container_width=True)

# SECCIÓN 4 — HALLAZGOS DEL EDA (INTERACTIVO)
elif st.session_state.sec == 3:

    st.markdown("## Hallazgos del EDA — Análisis Interactivo")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 ¿Quién usa el seguro?",
        "💰 ¿Cuánto cuesta?",
        "🏥 ¿De qué se enferma?",
        "📍 ¿Dónde?",
        "🔗 Correlaciones",
    ])

    # ── TAB 1: ¿QUIÉN? ────────────────────────────────────────
    with tab1:
        grupos = ["0-17", "18-30", "31-45", "46-60", "61-75", "76+"]
        conteo = [57915, 38036, 86612, 50767, 23163, 4772]
        usaron = [45524, 27454, 75612, 46908, 22283, 4704]
        tasa   = [round(u/t*100, 1) for u, t in zip(usaron, conteo)]

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("#### Tasa de uso por grupo de edad")
            fig = go.Figure(go.Bar(
                x=grupos, y=tasa,
                marker_color=["#0D9488" if t > 90 else "#5EEAD4" for t in tasa],
                text=[f"{t}%" for t in tasa], textposition="outside",
            ))
            fig.update_layout(
                xaxis_title="Grupo de edad", yaxis_title="% que usó el seguro",
                plot_bgcolor="white", paper_bgcolor="white",
                height=350, margin=dict(t=20, b=10),
                yaxis=dict(gridcolor="#F1F5F9", range=[0, 110], tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                xaxis=dict(tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                font=dict(color="#1E293B")
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Tasa de uso por sexo y grupo de edad")
            fig2 = go.Figure()
            fig2.add_bar(name="Femenino",  x=grupos,
                         y=[76.2, 75.1, 88.4, 93.2, 97.1, 98.2],
                         marker_color="#D85A30", opacity=0.85)
            fig2.add_bar(name="Masculino", x=grupos,
                         y=[80.5, 68.9, 86.1, 91.8, 95.4, 97.4],
                         marker_color="#378ADD", opacity=0.85)
            fig2.update_layout(
                barmode="group",
                xaxis_title="Grupo de edad", yaxis_title="% que usó el seguro",
                plot_bgcolor="white", paper_bgcolor="white",
                height=350, margin=dict(t=20, b=10),
                yaxis=dict(gridcolor="#F1F5F9", range=[0, 110], tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                xaxis=dict(tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                font=dict(color="#1E293B")
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Pirámide
        st.markdown("#### Pirámide poblacional — Femenino vs Masculino")
        fig_pir = go.Figure()
        fig_pir.add_bar(name="Femenino",
                        x=[-v for v in [27921, 19843, 47291, 28023, 12831, 2714]],
                        y=grupos, orientation="h", marker_color="#D85A30", opacity=0.85)
        fig_pir.add_bar(name="Masculino",
                        x=[29994, 18193, 39321, 22744, 10332, 2058],
                        y=grupos, orientation="h", marker_color="#378ADD", opacity=0.85)
        fig_pir.update_layout(
            barmode="overlay",
            xaxis=dict(
                title="← Femenino | Masculino →",
                tickvals=[-30000,-20000,-10000,0,10000,20000,30000],
                ticktext=["30K","20K","10K","0","10K","20K","30K"],
                tickfont=dict(color="#1E293B"),
                titlefont=dict(color="#1E293B")
            ),
            yaxis=dict(
                tickfont=dict(color="#1E293B"),
                titlefont=dict(color="#1E293B")
            ),
            plot_bgcolor="white", paper_bgcolor="white",
            height=300, margin=dict(t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#1E293B")),
            font=dict(color="#1E293B")
        )
        st.plotly_chart(fig_pir, use_container_width=True)

        st.markdown("""
            <div class="insight" style="color:#1E293B">
                <strong>Hallazgo 1:</strong> A mayor edad, mayor tasa de uso.
              El grupo 76+ usa el seguro el 97.8% de las veces vs 78.5% en niños (0-17).
              El grupo 31-45 es el más numeroso (86.612 personas) pero no el más costoso.
              Las mujeres en edad reproductiva (18-45) tienen mayor tasa de uso que los hombres.
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: ¿CUÁNTO CUESTA? ────────────────────────────────
    with tab2:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("#### Percentiles del costo total")
            pct_labels = ["P25", "P50\nMediana", "P75", "P90", "P95", "P99", "Promedio"]
            pct_vals   = [486463, 1512496, 4410960, 11717115, 20239760, 56674465, 5434389]
            pct_colors = ["#5EEAD4","#0D9488","#F97316","#DC2626","#991B1B","#450A0A","#7C3AED"]
            fig_pct = go.Figure(go.Bar(
                x=pct_labels, y=[v/1e6 for v in pct_vals],
                marker_color=pct_colors,
                text=[f"${v/1e6:.1f}M" for v in pct_vals], textposition="outside",
            ))
            fig_pct.update_layout(
                xaxis_title="Percentil", yaxis_title="Costo (Millones COP)",
                plot_bgcolor="white", paper_bgcolor="white",
                height=360, margin=dict(t=20, b=10),
                yaxis=dict(gridcolor="#F1F5F9", range=[0, 110], tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                xaxis=dict(tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                font=dict(color="#1E293B")
            )
            st.plotly_chart(fig_pct, use_container_width=True)

        with col2:
            st.markdown("#### Costo promedio por grupo de edad")
            costo_e = [1950000, 3200000, 4800000, 8900000, 15300000, 21700000]
            fig_ce = go.Figure(go.Bar(
                x=grupos, y=[v/1e6 for v in costo_e],
                marker_color=["#5EEAD4","#14B8A6","#0D9488","#F97316","#DC2626","#991B1B"],
                text=[f"${v/1e6:.1f}M" for v in costo_e], textposition="outside",
            ))
            fig_ce.update_layout(
                xaxis_title="Grupo de edad", yaxis_title="Costo promedio (M COP)",
                plot_bgcolor="white", paper_bgcolor="white",
                height=360, margin=dict(t=20, b=10),
                yaxis=dict(gridcolor="#F1F5F9", range=[0, 110], tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                xaxis=dict(tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                font=dict(color="#1E293B")
            )
            st.plotly_chart(fig_ce, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        for col, (val, lab, color) in zip([c1, c2, c3], [
            ("$5.4M",  "Costo PROMEDIO\n(quienes reclamaron)", "#7C3AED"),
            ("$1.5M",  "Costo MEDIANO\n(quienes reclamaron)",  "#0D9488"),
            ("x3.6",   "Brecha promedio/mediana\n→ usar log1p", "#F59E0B"),
        ]):
            with col:
                border = "border:2px solid #F59E0B;" if "3.6" in val else ""
                st.markdown(f"""
                <div class="metric-card" style="{border}">
                  <div class="metric-val" style="color:{color}">{val}</div>
                  <div class="metric-lab">{lab}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight" style="color:#1E293B">
          <strong>Hallazgo 2:</strong> La distribución de costos es muy asimétrica.
          La mayoría gasta poco, pero unos pocos generan costos enormes.
          La brecha de x3.6 lo confirma
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 3: ¿DE QUÉ? ───────────────────────────────────────
    with tab3:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("#### Costo promedio por tipo de servicio")
            servicios = ["TRAT. CÁNCER","HOSP. COMPLEJAS","CONSULTAS ESPEC.",
                         "HOSP. SIMPLES","CIRUGÍA","LABORATORIO",
                         "EXÁMENES MED.","CONSULTA MÉDICA"]
            costos_sv  = [69931159,49657898,36203415,26485802,
                          13525463,8094557,6835546,5466336]
            n_usu      = [1970,7065,5086,10160,49380,11061,162377,210533]

            fig_sv = go.Figure(go.Bar(
                y=servicios, x=[v/1e6 for v in costos_sv], orientation="h",
                marker_color=["#991B1B","#DC2626","#F97316","#FB923C",
                              "#7C3AED","#0D9488","#14B8A6","#0891B2"],
                text=[f"${v/1e6:.1f}M | n={n:,}" for v, n in zip(costos_sv, n_usu)],
                textposition="outside",
            ))
            fig_sv.update_layout(
                xaxis_title="Costo promedio (Millones COP)",
                plot_bgcolor="white", paper_bgcolor="white",
                height=380, margin=dict(t=10, b=10, r=160),
                xaxis=dict(gridcolor="#F1F5F9", range=[0, 75], tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                yaxis=dict(tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                font=dict(color="#1E293B")
            )
            st.plotly_chart(fig_sv, use_container_width=True)

        with col2:
            st.markdown("#### Prevalencia de condiciones preexistentes")
            conds  = ["HIPERTENSIÓN","DIABETES","ENF. CARDIACA","ENF. PULMONAR","CÁNCER"]
            prev   = [10.67, 2.45, 1.88, 1.40, 0.75]
            ratios = [3.1, 4.2, 3.8, 4.9, 4.5]

            fig_cond = go.Figure(go.Bar(
                x=prev, y=conds, orientation="h",
                marker_color="#F97316", opacity=0.8,
                text=[f"{p:.2f}%" for p in prev], textposition="outside",
            ))
            fig_cond.update_layout(
                xaxis_title="% de asegurados con la condición",
                plot_bgcolor="white", paper_bgcolor="white",
                height=210, margin=dict(t=10, b=10, r=60),
                xaxis=dict(gridcolor="#F1F5F9", tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                yaxis=dict(tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                font=dict(color="#1E293B")
            )
            st.plotly_chart(fig_cond, use_container_width=True)

            st.markdown("#### Multiplicador de costo (con / sin condición)")
            fig_rat = go.Figure(go.Bar(
                x=ratios, y=conds, orientation="h",
                marker_color=["#DC2626" if r > 4 else "#F97316" for r in ratios],
                text=[f"{r:.1f}x más costoso" for r in ratios], textposition="outside",
            ))
            fig_rat.add_vline(x=1, line_dash="dash", line_color="black")
            fig_rat.update_layout(
                xaxis_title="Veces más costoso",
                plot_bgcolor="white", paper_bgcolor="white",
                height=210, margin=dict(t=10, b=10, r=170),
                xaxis=dict(gridcolor="#F1F5F9", tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                yaxis=dict(tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                font=dict(color="#1E293B")  
            )
            st.plotly_chart(fig_rat, use_container_width=True)

        st.markdown("""
        <div class="insight" style="color:#1E293B">
          <strong>Hallazgo 3:</strong>
          <strong>CONSULTA MÉDICA</strong> = más frecuente (210.533 usuarios) pero más barata ($5.5M).
          <strong>TRATAMIENTO DE CÁNCER</strong> = más cara ($69.9M) pero más rara (1.970 usuarios).
          Esta <strong>relación inversa frecuencia-severidad</strong>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 4: ¿DÓNDE? ────────────────────────────────────────
    with tab4:
        # Datos reales
        ciudades   = ["BOGOTA", "MEDELLIN", "CALI", "CARTAGENA", "SIN INFORMACION"]
        n_aseg     = [130007, 59008, 38759, 33038, 41]
        costo_prom = [4.89, 4.15, 4.29, 4.46, 1.72]  # En millones COP, redondeado a 2 decimales

        col1, col2 = st.columns(2, gap="large")
        with col1:
            fig_g1 = go.Figure(go.Bar(
                x=n_aseg, y=ciudades, orientation="h",
                marker_color="#0D9488",
                text=[f"{n:,}" for n in n_aseg], textposition="outside",
            ))
            fig_g1.update_layout(
                title="Asegurados por ciudad",
                xaxis_title="Número de asegurados",
                plot_bgcolor="white", paper_bgcolor="white",
                height=380, margin=dict(t=40, b=10, r=80),
                xaxis=dict(gridcolor="#F1F5F9", tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                yaxis=dict(tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                font=dict(color="#1E293B")
            )
            st.plotly_chart(fig_g1, use_container_width=True)

        with col2:
            fig_g2 = go.Figure(go.Bar(
                x=costo_prom, y=ciudades, orientation="h",
                marker_color=["#DC2626" if v > 5 else "#F97316" if v > 4.5 else "#0D9488"
                              for v in costo_prom],
                text=[f"${v:.2f}M" for v in costo_prom], textposition="outside",
            ))
            fig_g2.add_vline(x=4.89, line_dash="dash", line_color="gray",
                             annotation_text="Media $4.57M")
            fig_g2.update_layout(
                title="Costo promedio por ciudad (M COP)",
                xaxis_title="Costo promedio (Millones COP)",
                plot_bgcolor="white", paper_bgcolor="white",
                height=380, margin=dict(t=40, b=10, r=80),
                xaxis=dict(gridcolor="#F1F5F9", tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                yaxis=dict(tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
                font=dict(color="#1E293B")
            )
            st.plotly_chart(fig_g2, use_container_width=True)
        st.markdown("""
        <div class="insight" style="color:#1E293B">
          <strong>Hallazgo 4:</strong> Bogotá concentra el mayor volumen de asegurados (~50% del total) al mismo tiempo,
          Bogotá también tiene el mayor costo promedio por asegurado.
          </strong>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 5: CORRELACIONES ──────────────────────────────────
    with tab5:
        st.markdown("#### Variables más correlacionadas con total_pagado")

        variables = ["n_reclamaciones","n_eventos","num_condiciones",
                     "edad","HIPERTENSION","CANCER","ENF_PULMONAR",
                     "ENF_CARDIACA","DIABETES","meses_expuesto_total"]
        corrs = [0.3239,0.2881,0.1523,0.1376,0.1133,
                 0.0918,0.0795,0.0721,0.0665,0.0507]
        colores_c = ["#DC2626" if c > 0.2 else "#F97316" if c > 0.1 else "#0D9488"
                     for c in corrs]

        fig_corr = go.Figure(go.Bar(
            y=variables[::-1], x=corrs[::-1], orientation="h",
            marker_color=colores_c[::-1],
            text=[f"{c:.4f}" for c in corrs[::-1]], textposition="outside",
        ))
        fig_corr.update_layout(
            title="Correlación de Pearson con total_pagado",
            xaxis_title="Coeficiente de correlación",
            plot_bgcolor="white", paper_bgcolor="white",
            height=400, margin=dict(t=40, b=10, r=80),
            xaxis=dict(tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=dict(tickfont=dict(color="#1E293B"), titlefont=dict(color="#1E293B")),
            font=dict(color="#1E293B")
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("""
        <div class="card card-yellow">
          <h4 style="color:#92400E;margin:0 0 0.5rem"> Advertencia: </h4>
          <p style="font-size:0.88rem;color:#78350F;line-height:1.75;margin:0">
            <code>n_reclamaciones</code> y <code>n_eventos</code> tienen las correlaciones
            más altas (0.32 y 0.29) pero son historial de siniestros que
            <strong>NO existe cuando llega un cliente nuevo a cotizar</strong>.
            Incluirlas sería <strong>data leakage (Fuga de datos u aprendizaje de patrones irreales) </strong>.<br>
            Las variables válidas del modelo son: <code>edad</code>, <code>sexo</code>,
            <code>ciudad</code>, <code>condiciones preexistentes</code>,
            <code>meses_expuesto</code>.
          </p>
        </div>
        <div class="insight" style="margin-top:0.8rem;color:#1E293B">
             <strong>Hallazgo 5:</strong> Entre las variables válidas,
          <code>num_condiciones</code> (0.15) y <code>edad</code> (0.14) son los predictores
          más poderosos. Confirma la lógica actuarial: más edad + más condiciones = mayor costo.
        </div>
        """, unsafe_allow_html=True)

# SECCIÓN 5 — PRÓXIMOS PASOS Y DUDAS
elif st.session_state.sec == 4:
    st.divider()
    st.markdown("###  Dudas técnicas para el profe")

    dudas = [
        ("#7C3AED", "#F5F3FF", "1", "De acuerdo a la Distribución del target: ¿qué modelo usar?",
        "El costo total pagado (total_pagado) tiene una brecha grande entre promedio y mediana (x3.6)", 
        "¿Qué tipo de modelo o transformación recomendaría para predecir este valor? Es mejor transformar el target con log1p y usar modelos tradicionales como regresión lineal o Random Forest, o cúal otro modelo?"),

    ]

    for color, bg, num, titulo, pregunta, contexto in dudas:
        st.markdown(f"""
        <div style="background:{bg};border-left:5px solid {color};
                    border-radius:0 12px 12px 0;padding:1.2rem 1.4rem;margin-bottom:0.9rem">
          <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem">
            <span style="background:{color};color:white;border-radius:50%;
                         width:26px;height:26px;display:flex;align-items:center;
                         justify-content:center;font-weight:700;font-size:0.82rem;
                         flex-shrink:0">{num}</span>
            <h4 style="margin:0;color:{color}">{titulo}</h4>
          </div>
          <p style="font-size:0.88rem;color:#1E293B;line-height:1.75;margin:0 0 0.5rem">
            <strong>Pregunta:</strong> {pregunta}
          </p>
          <p style="font-size:0.8rem;color:#64748B;margin:0;line-height:1.6">
            <em>Contexto:</em> {contexto}
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("###  Variables seleccionadas para el modelo final")

    col_f, col_t, col_e = st.columns(3, gap="medium")
    with col_f:
        st.markdown("""
        <div class="card card-teal">
          <h4 style="color:#0D9488;margin:0 0 0.7rem"> Features (X) — 10 variables</h4>
          <div style="font-family:'Courier New',monospace;font-size:0.82rem;
                      color:#374151;line-height:2.3">
            edad<br>Sexo_Cd_limpio<br>CIUDAD_NORM<br>CANCER<br>DIABETES<br>
            ENF_CARDIACA<br>HIPERTENSION<br>ENF_PULMONAR<br>num_condiciones<br>
            meses_expuesto_total
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_t:
        st.markdown("""
        <div class="card card-purple">
          <h4 style="color:#7C3AED;margin:0 0 0.7rem"> Target (Y)</h4>
          <div style="font-family:'Courier New',monospace;font-size:1rem;
                      color:#7C3AED;font-weight:700;margin-bottom:0.6rem">
            total_pagado
        </div>
        """, unsafe_allow_html=True)

    with col_e:
        st.markdown("""
        <div class="card card-red">
          <h4 style="color:#DC2626;margin:0 0 0.7rem"> Excluidas — Data Leakage</h4>
          <div style="font-family:'Courier New',monospace;font-size:0.8rem;
                      color:#991B1B;line-height:2.1">
            n_reclamaciones<br>n_eventos<br>rec_CIRUGIA<br>rec_CONSULTA MEDICA<br>
            rec_EXAMENES MEDICOS<br>fecha_primera_reclamacion<br>
            uso_seguro<br>categoria_cobertura
          </div>
          <p style="font-size:0.78rem;color:#7F1D1D;margin:0.6rem 0 0;line-height:1.5">
            Solo existen DESPUÉS del siniestro. Un cliente nuevo no las tiene.
          </p>
        </div>
        """, unsafe_allow_html=True)