# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from fbprophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# Configuración de página
st.set_page_config(page_title="Kallpa Securities - Predicción de Activos", layout="wide")

# Título principal
st.title("Sistema de Predicción de Precios de Activos")
st.markdown("### MVP desarrollado exclusivamente para **Kallpa Securities SAB**")
st.markdown("""
**Kallpa Securities SAB** es una de las principales sociedades agentes de bolsa del Perú, especializada en intermediación bursátil, 
asesoría financiera y servicios para inversionistas minoristas e institucionales en la Bolsa de Valores de Lima (BVL).
Este MVP utiliza inteligencia artificial para predecir precios de activos clave del mercado peruano, integrando variables macroeconómicas críticas.
""")

# Login simple (sin base de datos)
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

def login():
    st.subheader("Acceso al Sistema - Kallpa Securities")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Iniciar Sesión"):
        if username in ["kallpa", "analista", "inversionista"] and password == "kallpa2025":
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Bienvenido, {username.upper()}!")
            st.experimental_rerun()
        else:
            st.error("Credenciales incorrectas. Pista: usuario = kallpa, contraseña = kallpa2025")

if not st.session_state.logged_in:
    login()
else:
    st.sidebar.success(f"Conectado como: {st.session_state.username.upper()}")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    # Sidebar - Selección de activo
    st.sidebar.header("Configuración de Predicción")
    activos_bvl = {
        "Southern Copper (SCCO)": "SCCO",
        "Buenaventura (BVN)": "BVN",
        "Credicorp": "BAP",
        "Volcan Clase B": "VOLCABC1.LM",
        "Unacem": "UNACEMC1.LM",
        "Ferreycorp": "FERREYC1.LM"
    }
    activo_nombre = st.sidebar.selectbox("Seleccione un activo", list(activos_bvl.keys()))
    ticker = activos_bvl[activo_nombre]

    dias_prediccion = st.sidebar.slider("Días a predecir", 7, 30, 14)

    # Variables macroeconómicas (simuladas - en producción: API BCRP)
    st.sidebar.subheader("Variables Macroeconómicas (Impacto Actual)")
    macro = {
        "Tipo de Cambio (USD/PEN)": st.sidebar.text_input("Tipo de Cambio", "3.78"),
        "Tasa BCRP (%)": st.sidebar.text_input("Tasa Referencia", "5.25"),
        "Precio Cobre (USD/lb)": st.sidebar.text_input("Cobre", "4.35"),
        "Inflación (%)": st.sidebar.text_input("Inflación Anual", "2.4")
    }

    if st.sidebar.button("Generar Predicción"):
        with st.spinner(f"Analizando {activo_nombre} con IA..."):
            # Cargar datos
            try:
                data = yf.download(ticker, period="2y", progress=False)
                if data.empty or len(data) < 100:
                    st.error("No se pudieron cargar datos suficientes para este activo.")
                    st.stop()
                df = data[['Close']].reset_index()
                df.columns = ['ds', 'y']

                # Entrenar modelo Prophet
                m = Prophet(daily_seasonality=True, yearly_seasonality=True)
                m.fit(df)

                future = m.make_future_dataframe(periods=dias_prediccion)
                forecast = m.predict(future)

                # Mostrar resultados
                st.success(f"Predicción generada para {activo_nombre}")

                col1, col2 = st.columns(2)
                with col1:
                    ultimo_precio = df['y'].iloc[-1]
                    prediccion_final = forecast['yhat'].iloc[-1]
                    variacion = ((prediccion_final - ultimo_precio) / ultimo_precio) * 100
                    st.metric("Precio Actual", f"S/ {ultimo_precio:.2f}")
                    st.metric(f"Predicción en {dias_prediccion} días", f"S/ {prediccion_final:.2f}", f"{variacion:+.2f}%")

                with col2:
                    tendencia = "Alcista" if variacion > 0 else "Bajista"
                    color = "🟢" if variacion > 0 else "🔴"
                    st.markdown(f"### Tendencia Pronosticada: {color} **{tendencia}**")

                # Gráfico interactivo
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Histórico', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicción', line=dict(dash='dash', color='green')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Límite Superior', line=dict(color='lightgreen', dash='dot')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Límite Inferior', fill='tonexty', line=dict(color='lightcoral', dash='dot')))
                fig.update_layout(title=f"Predicción de {activo_nombre} - Kallpa Securities SAB", xaxis_title="Fecha", yaxis_title="Precio (PEN)")
                st.plotly_chart(fig, use_container_width=True)

                # Tabla de predicción
                st.subheader("Pronóstico Detallado")
                ultimos = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(dias_prediccion).copy()
                ultimos['ds'] = ultimos['ds'].dt.strftime('%d/%m/%Y')
                ultimos.rename(columns={'ds': 'Fecha', 'yhat': 'Predicción', 'yhat_lower': 'Mínimo', 'yhat_upper': 'Máximo'}, inplace=True)
                st.dataframe(ultimos.style.format("{:.2f}"), use_container_width=True)

            except Exception as e:
                st.error(f"Error al procesar el activo: {str(e)}")

    # Sección Q&A
    st.markdown("---")
    st.subheader("Preguntas Frecuentes - Kallpa Securities SAB")
    with st.expander("¿Qué es este sistema MVP?"):
        st.write("Es un prototipo funcional de inteligencia artificial para predecir precios de activos en la BVL, diseñado específicamente para Kallpa Securities SAB.")
    with st.expander("¿Qué modelo usa?"):
        st.write("Utiliza **Facebook Prophet**, un modelo de series temporales robusto y probado en mercados financieros.")
    with st.expander("¿Puedo confiar en las predicciones?"):
        st.write("Es una herramienta de apoyo a la decisión. Las predicciones son probabilísticas. Siempre combine con análisis fundamental y asesoría profesional de Kallpa.")
    with st.expander("¿Quién puede usarlo?"):
        st.write("Este MVP está diseñado para analistas, asesores y clientes de Kallpa Securities SAB.")
    with st.expander("¿Cómo contacto a Kallpa?"):
        st.write("Visita [www.kallpasab.com](https://www.kallpasab.com) o escribe a research@kallpasab.com")

    st.markdown("---")
    st.caption("MVP desarrollado por estudiantes de Ingeniería de Sistemas - UPC | Dedicado a Kallpa Securities SAB | 2025")
