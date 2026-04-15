# streamlit_app.py - DEMO FINAL - Gráficos visibles y estables
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Kallpa Securities - Demo", layout="wide")
st.title("🧠 Demo Predictiva - Kallpa Securities SAB")
st.markdown("### Sistema de Predicción de Precios para la BVL")

# Login simple
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("Acceso Demo")
    col1, col2 = st.columns(2)
    with col1:
        user = st.text_input("Usuario", placeholder="kallpa")
    with col2:
        pwd = st.text_input("Contraseña", type="password", placeholder="lstm2025")
    if st.button("Ingresar", type="primary"):
        if user == "kallpa" and pwd == "lstm2025":
            st.session_state.logged_in = True
            st.success("Acceso concedido")
            st.rerun()
        else:
            st.error("Credenciales incorrectas")
else:
    st.sidebar.success("✅ Sesión activa")

    # Configuración
    st.sidebar.header("Configuración de Demo")
    activos = {
        "Southern Copper (SCCO)": "SCCO",
        "Buenaventura (BVN)": "BVN",
        "Credicorp (BAP)": "BAP",
        "Volcan Clase B": "VOLCABC1.LM"
    }
    activo = st.sidebar.selectbox("Selecciona un activo", list(activos.keys()))
    symbol = activos[activo]

    modo = st.sidebar.selectbox("Modo de Modelo", [
        "LSTM Simulado",
        "LSTM + GRU",
        "Ensemble Completo"
    ])

    st.sidebar.subheader("Variables Macroeconómicas")
    tc = st.sidebar.slider("Tipo de Cambio USD/PEN", 3.5, 4.2, 3.78)
    tasa = st.sidebar.slider("Tasa BCRP (%)", 4.0, 8.0, 5.25)
    cobre = st.sidebar.slider("Precio Cobre USD/lb", 3.5, 5.5, 4.35)

    if st.sidebar.button("🚀 Generar Predicción", type="primary"):
        with st.spinner("Procesando datos y generando pronóstico..."):
            try:
                data = yf.download(symbol, period="2y", progress=False)
                if data.empty:
                    st.error("No se pudieron obtener datos para este activo.")
                    st.stop()

                close_col = 'Close' if 'Close' in data.columns else 'Adj Close'
                precios = pd.to_numeric(data[close_col], errors='coerce').dropna()
                
                if len(precios) < 30:
                    st.error("Datos insuficientes para generar predicción.")
                    st.stop()

                valores = precios.values.astype(float)
                precio_actual = float(valores[-1])

                # LSTM Simulado (robusto)
                window = min(60, len(valores))
                y_vent = valores[-window:]
                x = np.arange(window)

                if len(y_vent) < 2 or np.std(y_vent) < 0.01:
                    lstm_pred = precio_actual
                else:
                    grado = min(3, len(y_vent)-1)
                    coeffs = np.polyfit(x, y_vent, grado)
                    lstm_pred = float(np.polyval(coeffs, window))

                # GRU Simulado (EMA)
                ema = precio_actual
                for p in valores[-25:]:
                    ema = 0.2 * p + 0.8 * ema
                gru_pred = ema

                # ARIMA Simulado
                diff = np.diff(valores[-30:]) if len(valores) > 30 else np.array([0.0])
                tendencia = float(np.mean(diff))
                arima_pred = precio_actual + tendencia * 3

                # Fusión
                if modo == "Ensemble Completo":
                    base = 0.6 * lstm_pred + 0.25 * gru_pred + 0.15 * arima_pred
                elif modo == "LSTM + GRU":
                    base = 0.7 * lstm_pred + 0.3 * gru_pred
                else:
                    base = lstm_pred

                # Ajuste macro
                macro_impact = (tc - 3.78)*0.02 + (tasa - 5.25)*(-0.015) + (cobre - 4.35)*0.03
                prediccion_final = base * (1 + macro_impact)

                # Generar 14 días
                futuro = []
                actual = precio_actual
                for i in range(14):
                    paso = (prediccion_final - actual) / (14 - i)
                    ruido = np.random.normal(0, 0.01)
                    nuevo = actual + paso + ruido * actual
                    futuro.append(float(nuevo))
                    actual = nuevo

                variacion = ((futuro[-1] - precio_actual) / precio_actual) * 100

                # Resultados
                st.success(f"✅ Predicción generada con {modo}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Precio Actual", f"S/ {precio_actual:.2f}")
                col2.metric("Predicción 14 días", f"S/ {futuro[-1]:.2f}")
                col3.metric("Variación Esperada", f"{variacion:+.2f}%", delta=f"{variacion:+.2f}%")

                # Gráfico llamativo
                fechas_fut = [fechas[-1] + timedelta(days=i+1) for i in range(14)]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fechas[-90:], y=valores[-90:], name="Histórico", line=dict(color="#1f77b4", width=3)))
                fig.add_trace(go.Scatter(x=fechas_fut, y=futuro, name="Predicción", line=dict(color="#2ca02c", width=4, dash="dash"), marker=dict(size=8)))
                fig.update_layout(
                    title=f"{activo} - Pronóstico Kallpa Securities",
                    xaxis_title="Fecha",
                    yaxis_title="Precio (S/)",
                    height=550,
                    template="plotly_white",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Tabla simple
                df = pd.DataFrame({
                    "Fecha": [f.strftime("%d/%m") for f in fechas_fut],
                    "Predicción": [f"{p:.2f}" for p in futuro],
                    "Señal": ["🟢 COMPRA" if p > precio_actual*1.03 else "🔴 VENTA" if p < precio_actual*0.97 else "🟡 MANTENER" for p in futuro]
                })
                st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Error al procesar: {str(e)}")

else:  # Página Q&A
    st.title("Información y Q&A")
    st.subheader("Preguntas Frecuentes")
    with st.expander("¿Cómo funciona el modelo?"):
        st.write("Combina simulaciones de LSTM, GRU y ARIMA con ajuste por variables macroeconómicas peruanas.")
    with st.expander("¿Qué significa la variación?"):
        st.write("Es el porcentaje de cambio esperado en el precio del activo en 14 días.")
    with st.expander("Contacto Kallpa"):
        st.write("research@kallpasab.com | +51 1 219 0400")

st.caption("Demo MVP Kallpa Securities SAB | 2025")

st.caption("MVP Kallpa Securities SAB | 2025")
