# streamlit_app.py - MVP FINAL con Dashboard Storytelling (Gráfico siempre visible)
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
    st.subheader("Acceso Seguro – Kallpa Research")
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
        with st.spinner("Generando predicción híbrida..."):
            try:
                # Cargar datos
                data = yf.download(symbol, period="3y", progress=False)
                if data.empty:
                    st.error("No se encontraron datos")
                    st.stop()

                # Detectar columna de cierre automáticamente
                close_col = None
                for col in ['Close', 'CLOSE', 'Adj Close', 'close']:
                    if col in data.columns:
                        close_col = col
                        break
                if not close_col:
                    st.error("No se encontró columna de precios")
                    st.stop()

                precios = data[close_col].dropna()
                fechas = precios.index
                precios_values = precios.values.astype(float)

                if len(precios_values) < 60:
                    st.error("Datos insuficientes")
                    st.stop()

                ultimo_precio = float(precios_values[-1])

                # Simulación de modelos
                window = 60
                ventana = precios_values[-window:]
                x = np.arange(window)
                coeffs = np.polyfit(x, ventana, 3)
                lstm_pred = float(np.polyval(coeffs, window))

                ema = ultimo_precio
                for p in precios_values[-20:]:
                    ema = 0.2 * float(p) + 0.8 * ema
                gru_pred = ema

                diff = np.diff(precios_values[-30:])
                tendencia = np.mean(diff) if len(diff) > 0 else 0
                arima_pred = ultimo_precio + tendencia * 2

                # Fusión
                if modo == "Ensemble Completo":
                    base = 0.6 * lstm_pred + 0.25 * gru_pred + 0.15 * arima_pred
                elif modo == "LSTM + GRU Simulado":
                    base = 0.7 * lstm_pred + 0.3 * gru_pred
                else:
                    base = lstm_pred

                # Impacto macro
                macro_impact = (tc-3.78)*0.02 + (tasa-5.25)*(-0.015) + (cobre-4.35)*0.03
                prediccion_final = base * (1 + macro_impact)

                # Generar 14 días
                futuro = []
                actual = ultimo_precio
                for i in range(14):
                    paso = (prediccion_final - actual) / 14
                    ruido = np.random.normal(0, 0.008)
                    nuevo = actual + paso + ruido * actual
                    futuro.append(float(nuevo))
                    actual = nuevo

                # Resultados
                st.success(f"Predicción generada: {modo}")
                variacion = ((futuro[-1] - ultimo_precio) / ultimo_precio) * 100

                col1, col2, col3 = st.columns(3)
                col1.metric("Precio Actual", f"S/ {ultimo_precio:.2f}")
                col2.metric("Predicción 14d", f"S/ {futuro[-1]:.2f}")
                col3.metric("Variación", f"{variacion:+.2f}%", delta=f"{variacion:+.2f}%")

                # === DASHBOARD STORYTELLING CON GRÁFICO SIEMPRE VISIBLE ===
                st.markdown("### Análisis Predictivo Detallado")
                st.markdown(f"**Activo seleccionado:** {activo}")
                st.markdown(f"**Modelo utilizado:** {modo}")
                st.markdown(f"**Impacto macroeconómico neto:** {macro_impact:+.2%}")

                # Gráfico con datos limpios y fallback
                try:
                    num_hist = min(90, len(precios_values))
                    fechas_hist = fechas[-num_hist:]
                    precios_hist = precios_values[-num_hist:]

                    fechas_futuras = pd.date_range(start=fechas[-1] + timedelta(days=1), periods=14, freq='B')  # Días hábiles

                    fig = go.Figure()

                    # Histórico
                    fig.add_trace(go.Scatter(
                        x=fechas_hist,
                        y=precios_hist,
                        mode='lines',
                        name='Histórico',
                        line=dict(color="#1f77b4", width=3)
                    ))

                    # Predicción
                    fig.add_trace(go.Scatter(
                        x=fechas_futuras,
                        y=futuro,
                        mode='lines+markers',
                        name='Predicción',
                        line=dict(color="#2ca02c", width=3, dash='dash'),
                        marker=dict(size=6)
                    ))

                    # Banda de confianza
                    sup = [p * 1.06 for p in futuro]
                    inf = [p * 0.94 for p in futuro]
                    fig.add_trace(go.Scatter(x=fechas_futuras, y=sup, line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=fechas_futuras, y=inf, fill='tonexty',
                                           fillcolor='rgba(0,150,0,0.1)', line=dict(width=0), name="Confianza ±6%"))

                    fig.update_layout(
                        title=f"Evolución y Pronóstico - {activo}",
                        xaxis_title="Fecha",
                        yaxis_title="Precio (S/)",
                        height=550,
                        template="simple_white",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as graph_error:
                    st.warning("Gráfico temporalmente no disponible. La predicción numérica es válida.")
                    st.write(f"Debug: {graph_error}")

                # Tabla detallada
                df_futuro = pd.DataFrame({
                    "Fecha": [f.strftime("%d/%m/%Y") for f in fechas_futuras],
                    "Predicción (S/)": [f"{p:.2f}" for p in futuro],
                    "Señal": ["COMPRA" if p > ultimo_precio*1.03 else "VENTA" if p < ultimo_precio*0.97 else "MANTENER" for p in futuro]
                })
                st.dataframe(df_futuro, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

st.caption("MVP Kallpa Securities SAB | 2025")
