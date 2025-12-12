# streamlit_app.py - MVP FINAL 100% FUNCIONAL en Streamlit Cloud
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# Configuración
st.set_page_config(page_title="Kallpa Securities - Predicción IA", layout="wide")
st.title("Sistema Predictivo Híbrido - Kallpa Securities SAB")
st.markdown("PI1")



# Login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    col1, col2 = st.columns(2)
    with col1:
        user = st.text_input("Usuario")
    with col2:
        pwd = st.text_input("Contraseña", type="password")
    if st.button("Ingresar"):
        if user == "kallpa" and pwd == "lstm2025":
            st.session_state.logged_in = True
            st.success("Acceso concedido - Kallpa Securities SAB")
            st.rerun()
        else:
            st.error("Credenciales incorrectas")
else:
    st.sidebar.success("Sesión activa")
    if st.sidebar.button("Cerrar sesión"):
        st.session_state.logged_in = False
        st.rerun()

    # Configuración
    st.sidebar.header("Configuración")
    activos = {
        "Southern Copper (SCCO)": "SCCO",
        "Buenaventura (BVN)": "BVN",
        "Credicorp (BAP)": "BAP"
    }
    activo = st.sidebar.selectbox("Activo", list(activos.keys()))
    symbol = activos[activo]

    modo = st.sidebar.selectbox("Modo de Fusión", [
        "LSTM Simulado",
        "LSTM + GRU Simulado",
        "Ensemble Completo"
    ])

    # Macros
    st.sidebar.subheader("Variables Macroeconómicas")
    tc = st.sidebar.slider("Tipo Cambio", 3.5, 4.2, 3.78)
    tasa = st.sidebar.slider("Tasa BCRP (%)", 4.0, 8.0, 5.25)
    cobre = st.sidebar.slider("Cobre USD/lb", 3.5, 5.5, 4.35)
    inflacion = st.sidebar.slider("Inflación (%)", 1.5, 4.0, 2.4)

    # Sprint 2: Integración y Capacitación
    st.sidebar.header("Sprint 2: Funcionalidades")
    if st.sidebar.button("Generar Reporte Personalizado"):
        st.info("Reporte generado: Predicciones enviadas por email (simulado).")
    if st.sidebar.button("Iniciar Capacitación"):
        st.info("Sesión de capacitación iniciada: Tutorial interactivo para Research (simulado).")

    if st.sidebar.button("Generar Predicción"):
        with st.spinner("Procesando con modelo híbrido..."):
            try:
                # Datos
                data = yf.download(symbol, period="3y", progress=False)
                if len(data) < 100:
                    st.error("Datos insuficientes")
                    st.stop()

                df = data['Close'].reset_index()
                df['Date'] = pd.to_datetime(df['Date'])
                precios = df['Close'].values

                # Simulación LSTM (polinómica con memoria)
                window = 60
                lstm_sim = []
                for i in range(window, len(precios)):
                    x = np.arange(window)
                    y = precios[i-window:i]
                    poly_coeffs = np.polyfit(x, y, 2)
                    lstm_sim.append(np.polyval(poly_coeffs, window))
                lstm_mean = np.mean(lstm_sim[-14:] if len(lstm_sim) > 14 else lstm_sim) if lstm_sim else precios[-1]

                # Simulación GRU (EMA)
                alpha = 0.15
                gru_sim = np.zeros(len(precios))
                gru_sim[0] = precios[0]
                for i in range(1, len(precios)):
                    gru_sim[i] = alpha * precios[i] + (1 - alpha) * gru_sim[i-1]
                gru_mean = np.mean(gru_sim[-14:] if len(gru_sim) > 14 else gru_sim) if gru_sim.size > 0 else precios[-1]

                # Simulación ARIMA (tendencia lineal con autocorrelación)
                arima_sim = precios.copy()
                for i in range(1, len(arima_sim)):
                    arima_sim[i] = 0.8 * arima_sim[i-1] + 0.2 * (precios[i] - precios[i-1])
                arima_mean = np.mean(arima_sim[-14:] if len(arima_sim) > 14 else arima_sim) if arima_sim.size > 0 else precios[-1]

                # Fusión Ensemble
                if modo == "Ensemble Completo":
                    base_pred = 0.6 * lstm_mean + 0.25 * gru_mean + 0.15 * arima_mean
                elif modo == "LSTM + GRU Simulado":
                    base_pred = 0.7 * lstm_mean + 0.3 * gru_mean
                else:
                    base_pred = lstm_mean

                # Predicción futura (14 días)
                ultimo = precios[-1]
                futuro = []
                for i in range(14):
                    pred = base_pred * (1 + np.random.normal(0, 0.005) * (i+1))
                    macro_impact = (tc-3.78)*0.02 + (tasa-5.25)*(-0.015) + (cobre-4.35)*0.03 + (inflacion-2.4)*(-0.006)
                    pred = pred * (1 + macro_impact)
                    futuro.append(pred)
                    ultimo = pred

                # Resultados
                st.success(f"Predicción generada: {modo}")
                variacion = ((futuro[-1] - precios[-1]) / precios[-1]) * 100

                col1, col2, col3 = st.columns(3)
                col1.metric("Precio Actual", f"S/ {precios[-1]:.2f}")
                col2.metric("Predicción 14d", f"S/ {futuro[-1]:.2f}")
                col3.metric("Variación", f"{variacion:+.2f}%")

                # Gráfico
                fechas_fut = [df['Date'].iloc[-1] + timedelta(days=i+1) for i in range(14)]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'].tail(60), y=precios[-60:], name="Histórico"))
                fig.add_trace(go.Scatter(x=fechas_fut, y=futuro, name="Predicción Híbrida", line=dict(dash="dash")))
                fig.update_layout(title=f"{activo} - Kallpa Securities SAB")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

# Q&A
with st.expander("¿Cómo funciona la fusión híbrida?"):
    st.write("Modelo híbrido combina simulación de redes neuronales con métodos estadísticos para robustez en mercados peruanos.")

st.caption("MVP Kallpa Securities SAB 2025")
