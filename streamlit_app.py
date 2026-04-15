import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# ==============================
# CONFIGURACIÓN GLOBAL
# ==============================
st.set_page_config(
    page_title="Kallpa Securities - Dashboard BVL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# FUNCIONES CORE
# ==============================

def obtener_datos(symbol):
    data = yf.download(symbol, period="3y", progress=False)
    if data.empty:
        return None, None, None

    close_col = next((c for c in ['Close','CLOSE','Adj Close','close'] if c in data.columns), None)
    if not close_col:
        return None, None, None

    precios = data[close_col].dropna().values
    fechas = data.index

    return data, precios, fechas


def modelo_prediccion(precios, modo, tc, tasa, cobre):
    window = 60
    ultimo = float(precios[-1])

    # LSTM simulado (polinómico)
    ventana = precios[-window:]
    x = np.arange(window)
    coeffs = np.polyfit(x, ventana, 3)
    lstm = float(np.polyval(coeffs, window))

    # GRU (EMA)
    ema = ultimo
    for p in precios[-20:]:
        ema = 0.2 * float(p) + 0.8 * ema

    # ARIMA simplificado
    diff = np.diff(precios[-30:])
    tendencia = np.mean(diff) if len(diff) > 0 else 0
    arima = ultimo + tendencia * 2

    # Ensemble
    if modo == "Ensemble Completo":
        base = 0.6*lstm + 0.25*ema + 0.15*arima
    elif modo == "LSTM + GRU Simulado":
        base = 0.7*lstm + 0.3*ema
    else:
        base = lstm

    # Impacto macro
    impacto = (tc-3.78)*0.02 + (tasa-5.25)*(-0.015) + (cobre-4.35)*0.03
    final = base * (1 + impacto)

    return ultimo, final, impacto


def generar_futuro(ultimo, pred):
    futuro = []
    actual = ultimo

    for _ in range(14):
        paso = (pred - actual)/14
        ruido = np.random.normal(0, 0.008)
        nuevo = actual + paso + ruido * actual
        futuro.append(float(nuevo))
        actual = nuevo

    return futuro


# ==============================
# UI SIDEBAR
# ==============================

st.sidebar.title("⚙️ Configuración")

activos = {
    "Southern Copper": "SCCO",
    "Buenaventura": "BVN",
    "Credicorp": "BAP",
    "Volcan": "VOLCABC1.LM"
}

activo = st.sidebar.selectbox("Activo", activos.keys())
symbol = activos[activo]

modo = st.sidebar.selectbox("Modelo", [
    "LSTM Simulado",
    "LSTM + GRU Simulado",
    "Ensemble Completo"
])

st.sidebar.markdown("### Variables Macro")

tc = st.sidebar.slider("Tipo de Cambio", 3.5, 4.2, 3.78)
tasa = st.sidebar.slider("Tasa BCRP (%)", 4.0, 8.0, 5.25)
cobre = st.sidebar.slider("Cobre USD/lb", 3.5, 5.5, 4.35)

run = st.sidebar.button("🚀 Ejecutar Predicción")

# ==============================
# HEADER
# ==============================

st.title("📊 Dashboard Predictivo BVL")
st.caption("Modelo híbrido de series temporales + variables macroeconómicas")

# ==============================
# EJECUCIÓN
# ==============================

if run:
    with st.spinner("Procesando modelo..."):

        data, precios, fechas = obtener_datos(symbol)

        if data is None or len(precios) < 60:
            st.error("Datos insuficientes")
            st.stop()

        ultimo, pred_final, impacto = modelo_prediccion(precios, modo, tc, tasa, cobre)
        futuro = generar_futuro(ultimo, pred_final)

        variacion = ((futuro[-1] - ultimo)/ultimo)*100

        # ==============================
        # KPIs
        # ==============================

        col1, col2, col3 = st.columns(3)

        col1.metric("Precio Actual", f"S/ {ultimo:.2f}")
        col2.metric("Predicción (14d)", f"S/ {futuro[-1]:.2f}")
        col3.metric("Variación Esperada", f"{variacion:+.2f}%")

        # ==============================
        # INSIGHT EJECUTIVO
        # ==============================

        tendencia = "ALCISTA 📈" if variacion > 0 else "BAJISTA 📉"

        st.markdown(f"""
        ### 📌 Insight Ejecutivo
        - Activo: **{activo}**
        - Tendencia proyectada: **{tendencia}**
        - Impacto macroeconómico: **{impacto:+.2%}**
        
        **Interpretación:** El modelo sugiere una dinámica influenciada principalmente por tipo de cambio, tasa BCRP y precio del cobre.
        """)

        # ==============================
        # GRÁFICO
        # ==============================

        fig = go.Figure()

        hist = data[-90:]

        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name="Histórico"
        ))

        fechas_fut = [fechas[-1] + timedelta(days=i+1) for i in range(14)]

        fig.add_trace(go.Scatter(
            x=fechas_fut,
            y=futuro,
            mode='lines+markers',
            name="Predicción",
            line=dict(dash="dash")
        ))

        fig.update_layout(
            height=550,
            template="plotly_dark",
            title=f"Proyección de {activo}"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ==============================
        # TABLA
        # ==============================

        df = pd.DataFrame({
            "Fecha": fechas_fut,
            "Precio": futuro
        })

        st.dataframe(df, use_container_width=True)

