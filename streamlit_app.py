# streamlit_app.py - Versión FINAL CORREGIDA Y EXPLICADA
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
st.markdown("### MVP Avanzado | Simulación LSTM + GRU + ARIMA")

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
            st.success("Acceso concedido")
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
        "Credicorp (BAP)": "BAP",
        "Volcan B": "VOLCABC1.LM"
    }
    activo = st.sidebar.selectbox("Activo", list(activos.keys()))
    symbol = activos[activo]

    modo = st.sidebar.selectbox("Modo de Fusión", [
        "LSTM Simulado",
        "LSTM + GRU Simulado",
        "Ensemble Completo"
    ])

    # Variables macro
    st.sidebar.subheader("Variables Macroeconómicas")
    tc = st.sidebar.slider("Tipo Cambio", 3.5, 4.2, 3.78)
    tasa = st.sidebar.slider("Tasa BCRP (%)", 4.0, 8.0, 5.25)
    cobre = st.sidebar.slider("Cobre USD/lb", 3.5, 5.5, 4.35)

    if st.sidebar.button("Generar Predicción"):
        with st.spinner("Cargando datos y generando predicción..."):
            try:
                # CORREGIDO: Ahora detecta automáticamente la columna de cierre
                data = yf.download(symbol, period="3y", progress=False)
                
                # Esto es lo que fallaba: ahora buscamos la columna correcta
                close_col = None
                for col in ['Close', 'CLOSE', 'Adj Close', 'close']:
                    if col in data.columns:
                        close_col = col
                        break
                
                if close_col is None or data.empty:
                    st.error("No se encontraron datos de precios para este activo.")
                    st.stop()

                # Ahora sí: usamos la columna correcta
                precios = data[close_col].dropna().values
                fechas = data.index[-60:]  # Últimos 60 días

                if len(precios) < 60:
                    st.error("Datos insuficientes (menos de 60 días)")
                    st.stop()

                # === SIMULACIÓN DE MODELOS (como en tu tesis) ===

                # 1. LSTM simulado: regresión polinómica con ventana de 60 días
                def lstm_simulado(precios, ventana=60):
                    ultimos = precios[-ventana:]
                    x = np.arange(ventana)
                    coeffs = np.polyfit(x, ultimos, 3)
                    return np.polyval(coeffs, ventana)  # predicción del siguiente día

                lstm_pred = lstm_simulado(precios)

                # 2. GRU simulado: EMA (más rápido que LSTM)
                ema = precios[-1]
                for p in precios[-30:]:
                    ema = 0.2 * p + 0.8 * ema
                gru_pred = ema

                # 3. ARIMA simulado: tendencia + autocorrelación
                diff = np.diff(precios[-30:])
                arima_trend = np.mean(diff) if len(diff) > 0 else 0
                arima_pred = precios[-1] + arima_trend * 1.5

                # === FUSIÓN FINAL ===
                if modo == "Ensemble Completo":
                    prediccion_base = 0.6 * lstm_pred + 0.25 * gru_pred + 0.15 * arima_pred
                elif modo == "LSTM + GRU Simulado":
                    prediccion_base = 0.7 * lstm_pred + 0.3 * gru_pred
                else:
                    prediccion_base = lstm_pred

                # Aplicar impacto macroeconómico (como en tu tesis)
                macro_impact = (
                    (tc - 3.78) * 0.02 +
                    (tasa - 5.25) * (-0.015) +
                    (cobre - 4.35) * 0.03
                )
                prediccion_final = prediccion_base * (1 + macro_impact)

                # Generar 14 días de predicción
                futuro = []
                ultimo = precios[-1]
                for i in range(14):
                    variacion_diaria = (prediccion_final - ultimo) / 14
                    ruido = np.random.normal(0, 0.008)
                    nuevo = ultimo + variacion_diaria * (i + 1) + ruido * ultimo
                    futuro.append(nuevo)

                # === RESULTADOS ===
                st.success(f"Predicción generada con {modo}")
                
                variacion_total = ((futuro[-1] - precios[-1]) / precios[-1]) * 100

                col1, col2, col3 = st.columns(3)
                col1.metric("Precio Actual", f"S/ {precios[-1]:.2f}")
                col2.metric("Predicción 14d", f"S/ {futuro[-1]:.2f}")
                col3.metric("Variación Esperada", f"{variacion_total:+.2f}%", 
                          delta=f"{variacion_total:+.2f}%")

                # Gráfico
                fechas_futuras = [fechas[-1] + timedelta(days=i+1) for i in range(14)]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fechas, y=precios[-60:], name="Histórico", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=fechas_futuras, y=futuro, name="Predicción Híbrida", 
                                       line=dict(color="green", dash="dash", width=3)))
                fig.update_layout(title=f"{activo} - Kallpa Securities SAB", height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Tabla
                df_futuro = pd.DataFrame({
                    "Fecha": [f.strftime("%d/%m") for f in fechas_futuras],
                    "Predicción": [f"S/ {p:.2f}" for p in futuro],
                    "Señal": ["COMPRA" if p > precios[-1]*1.03 else "VENTA" if p < precios[-1]*0.97 else "MANTENER" for p in futuro]
                })
                st.dataframe(df_futuro, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

st.caption("MVP Kallpa Securities SAB | 2025")
