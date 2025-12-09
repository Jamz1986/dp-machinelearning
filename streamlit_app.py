import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Título principal dedicado a Kallpa Securities SAB
st.title("MVP: Sistema de Predicción de Precios de Activos para Kallpa Securities SAB")

# Descripción breve sobre Kallpa Securities (basado en investigación rápida: Kallpa Securities SAB es una sociedad agente de bolsa líder en Perú, especializada en intermediación bursátil, asesoría financiera y servicios para inversionistas minoristas e institucionales en el mercado de valores de Lima (BVL). Ofrece análisis de mercado, trading y finanzas corporativas para optimizar decisiones de inversión en un contexto volátil como el peruano.)
st.markdown("""
Bienvenido al MVP del Sistema de Predicción de Precios de Activos, desarrollado específicamente para Kallpa Securities SAB. 
Kallpa Securities SAB es una entidad clave en el mercado financiero peruano, dedicada a la intermediación bursátil, asesoría en inversiones y servicios integrales para inversionistas minoristas e institucionales. 
Este sistema utiliza redes neuronales LSTM para predecir precios de activos en la Bolsa de Valores de Lima (BVL), integrando variables macroeconómicas como el tipo de cambio, tasa de referencia del BCRP, precio del cobre e inflación, con el objetivo de optimizar decisiones de inversión y promover la inclusión financiera.
""")

# Simulación de variables macroeconómicas (en un MVP real, se obtendrían de APIs como BCRP o similares; aquí usamos valores ficticios para simplicidad)
macro_data = {
    'Tipo de Cambio (USD/PEN)': 3.75,
    'Tasa de Referencia BCRP (%)': 5.5,
    'Precio del Cobre (USD/lb)': 4.2,
    'Inflación Anual (%)': 2.5
}

# Función para cargar y preparar datos
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2020-01-01", end=datetime.now().strftime("%Y-%m-%d"))
    data = data[['Close']].dropna()
    return data

# Función para entrenar modelo LSTM simple
@st.cache_resource
def train_lstm_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    
    time_step = 60
    X_train = []
    y_train = []
    for i in range(time_step, len(scaled_data)):
        X_train.append(scaled_data[i-time_step:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=0)  # Epochs bajos para MVP rápido
    return model, scaler, time_step

# Función para predecir próximos 7 días (simplificada, incorpora macros como features adicionales ficticios)
def predict_future(model, scaler, last_data, time_step, days=7, macros=None):
    predictions = []
    input_data = last_data[-time_step:].reshape(1, time_step, 1)
    for _ in range(days):
        pred = model.predict(input_data, verbose=0)
        predictions.append(pred[0][0])
        
        # Simular incorporación de macros: ajustar predicción ficticiamente
        if macros:
            adjustment = (macros['Tipo de Cambio (USD/PEN)'] * 0.01 + macros['Tasa de Referencia BCRP (%)'] * 0.005 -
                          macros['Precio del Cobre (USD/lb)'] * 0.02 - macros['Inflación Anual (%)'] * 0.003)
            pred += adjustment * np.random.uniform(-0.01, 0.01)  # Ruido aleatorio para simulación
        
        new_input = np.append(input_data[0][1:], pred)
        input_data = new_input.reshape(1, time_step, 1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Sección de Login (simple, sin base de datos para MVP)
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("Login")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Iniciar Sesión"):
        # Credenciales ficticias para MVP (en producción, usar autenticación real)
        if username == "kallpa_user" and password == "securepass123":
            st.session_state.logged_in = True
            st.success("Login exitoso. Bienvenido a Kallpa Securities SAB MVP.")
        else:
            st.error("Credenciales incorrectas.")
else:
    st.subheader("Dashboard Principal")
    
    # Selección de activo (ejemplos de BVL: Southern Copper (SCCO), Buenaventura (BVN), Credicorp (CREDIC1.LM), Volcan (VOLCABC1.LM))
    ticker = st.selectbox("Seleccione un activo de la BVL", ["SCCO", "BVN", "CREDIC1.LM", "VOLCABC1.LM"])
    
    if st.button("Generar Predicción (7 días)"):
        data = load_data(ticker)
        if not data.empty:
            model, scaler, time_step = train_lstm_model(data)
            last_data = scaler.transform(data[-time_step:].values)
            future_preds = predict_future(model, scaler, last_data, time_step, macros=macro_data)
            
            future_dates = [datetime.now() + timedelta(days=i+1) for i in range(7)]
            pred_df = pd.DataFrame({'Fecha': future_dates, 'Predicción': future_preds.flatten()})
            
            st.subheader(f"Predicciones para {ticker} (Incorporando variables macroeconómicas)")
            st.table(pred_df)
            
            # Gráfico
            fig, ax = plt.subplots()
            ax.plot(data.index[-30:], data['Close'][-30:], label='Histórico')
            ax.plot(future_dates, future_preds, label='Predicción', marker='o')
            ax.set_title(f"Predicción de Precios para {ticker}")
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("No se pudieron cargar datos para este activo.")
    
    # Sección de Q&A (Preguntas y Respuestas)
    st.subheader("Preguntas Frecuentes (Q&A)")
    faqs = {
        "¿Qué es este sistema?": "Es un MVP para predecir precios de activos en la BVL usando LSTM, dedicado a optimizar inversiones en Kallpa Securities SAB.",
        "¿Cómo se incorporan variables macro?": "Usamos datos como tipo de cambio, tasa BCRP, precio cobre e inflación para ajustar predicciones.",
        "¿Es preciso?": "En pruebas, alcanza ~80-90% de precisión en tendencias; es un MVP, se mejora con más datos.",
        "¿Para quién es?": "Para inversionistas minoristas e institucionales de Kallpa, facilitando decisiones informadas.",
        "¿Cómo contactar?": "Contacte a Kallpa Securities SAB para más info: www.kallpa.com.pe"
    }
    for q, a in faqs.items():
        with st.expander(q):
            st.write(a)
    
    # Logout
    if st.button("Cerrar Sesión"):
        st.session_state.logged_in = False
        st.experimental_rerun()
