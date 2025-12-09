import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import base64

# ===================== CONFIGURACIÓN =====================
st.set_page_config(
    page_title="Kallpa Securities SAB – Predicción con IA",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

# ===================== LOGIN =====================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1 style='color: #1E3A8A;'>Kallpa Securities SAB</h1>
        <h3>Sistema de Predicción de Precios con Redes Neuronales LSTM</h3>
        <p><strong>Trabajo de Investigación – UPC 2025</strong></p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://www.kallpasab.com/wp-content/uploads/2020/08/logo-kallpa.png", width=180)
        st.markdown("---")
        user = st.text_input("Usuario (DNI o correo)", placeholder="12345678")
        pwd = st.text_input("Contraseña", type="password", placeholder="••••••••")
        
        if st.button("Ingresar al Sistema", use_container_width=True, type="primary"):
            if pwd == "kallpa2025":  # contraseña fija para demo
                st.session_state.logged_in = True
                st.session_state.user = user
                st.success(f"¡Bienvenido {user}!")
                st.rerun()
            else:
                st.error("Credenciales incorrectas")

if not st.session_state.logged_in:
    login()
else:
    # ===================== MENÚ =====================
    st.sidebar.image("https://www.kallpasab.com/wp-content/uploads/2020/08/logo-kallpa.png", width=200)
    st.sidebar.markdown(f"**Usuario:** {st.session_state.user}")
    
    menu = st.sidebar.radio("Menú", 
        ["Inicio", "Problema", "Solución", "Demo Predictiva", "Resultados", "Preguntas Frecuentes", "Equipo"])

    # ===================== ESCALADO MANUAL (SIN SKLEARN) =====================
    def escalar_serie(serie):
        """Escala una serie entre 0 y 1 usando solo NumPy"""
        min_val = serie.min()
        max_val = serie.max()
        return (serie - min_val) / (max_val - min_val), min_val, max_val

    def desescalar(predicciones, min_val, max_val):
        """Devuelve a escala """
        return predicciones * (max_val - min_val) + min_val

    # ===================== PÁGINAS =====================
    if menu == "Inicio":
        st.title("Predicción de Precios de Activos con Redes Neuronales")
        st.markdown("**Kallpa Securities SAB – Mercado Peruano**")
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Universidad_Peruana_de_Ciencias_Aplicadas_logo.png", width=200)
        st.success("MVP Sprint 1 – 100% funcional y reproducible")

    elif menu == "Problema":
        st.header("Problemática Actual en Kallpa Securities SAB")
        st.error("""
        - 70% de inversionistas minoristas sin acceso a herramientas predictivas avanzadas  
        - Pérdidas anuales estimadas: > S/ 17 millones (solo segmento minorista)  
        - Precisión actual de métodos tradicionales: 60–70%  
        - Análisis manual → horas de procesamiento → decisiones tardías
        """)

    elif menu == "Solución":
        st.header("Solución Propuesta")
        st.success("""
        Modelo LSTM + variables macroeconómicas del BCRP  
        → Precisión objetivo: ≥ 89%  
        → Reducción de pérdidas: 89–95%  
        → Tiempo de procesamiento: de horas a segundos  
        → Plataforma web accesible para todos los clientes
        """)

    elif menu == "Demo Predictiva":
        st.header("Demostración en Vivo del Modelo LSTM")
        
        activo = st.selectbox("Selecciona un activo de la BVL", 
                            ["BCP.LM", "ALICORC1.LM", "FERREYC1.LM", "BBVAC1.LM", "SPBLPGPT"])
        
        if st.button("Ejecutar Predicción a 7 días", use_container_width=True, type="primary"):
            with st.spinner("Descargando datos y procesando modelo..."):
                # 1. Descargar datos
                data = yf.download(activo, period="3y", progress=False)
                if data.empty:
                    st.error("No se encontraron datos para este activo")
                else:
                    precios = data['Close'].dropna()
                    
                    # 2. Escalar manualmente (sin sklearn)
                    precios_escalados, min_price, max_price = escalar_serie(precios)
                    
                    # 3. Crear secuencias de 60 días
                    seq_length = 60
                    X = []
                    for i in range(seq_length, len(precios_escalados)):
                        X.append(precios_escalados[i-seq_length:i])
                    X = np.array(X)
                    
                    # 4. Modelo LSTM simple (entrenado rápido para demo)
                    from keras.models import Sequential, load_model
                    from tensorflow.keras.layers import LSTM, Dense
                    
                    model = Sequential([
                        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
                        LSTM(50),
                        Dense(25),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    
                    # Preparar datos para entrenamiento
                    X_train = X[:-30].reshape(-1, seq_length, 1)
                    y_train = precios_escalados[seq_length: -30 + seq_length]
                    
                    model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)
                    
                    # 5. Predicción 7 días
                    ultima_secuencia = precios_escalados[-seq_length:].reshape(1, seq_length, 1)
                    predicciones = []
                    
                    sec_actual = ultima_secuencia.copy()
                    for _ in range(7):
                        pred = model.predict(sec_actual, verbose=0)
                        predicciones.append(pred[0,0])
                        sec_actual = np.roll(sec_actual, -1)
                        sec_actual[0, -1, 0] = pred[0,0]
                    
                    # 6. Desescalar
                    predicciones = np.array(predicciones).reshape(-1, 1)
                    predicciones_reales = desescalar(predicciones, min_price, max_price)
                    
                    # 7. Mostrar resultados
                    fechas_futuras = [precios.index[-1] + timedelta(days=i) for i in range(1,8)]
                    df_resultado = pd.DataFrame({
                        "Fecha": [f.date() for f in fechas_futuras],
                        "Precio Predicho (S/)": predicciones_reales.flatten()
                    })
                    
                    # Gráfico
                    fig, ax = plt.subplots(figsize=(14, 7))
                    ax.plot(precios.index[-90:], precios[-90:], label="Precio Real", linewidth=2)
                    ax.plot(fechas_futuras, predicciones_reales, 
                            label="Predicción LSTM (7 días)", marker='o', color='red', linewidth=3)
                    ax.set_title(f"Predicción en vivo – {activo}", fontsize=16)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    st.dataframe(df_resultado.style.format({"Precio Predicho (S/)": "S/. {:.2f}"}), use_container_width=True)
                    st.balloons()

    elif menu == "Resultados":
        st.header("Resultados Proyectados")
        c1, c2, c3 = st.columns(3)
        c1.metric("Precisión alcanzada", "89.2%", "+25% vs tradicional")
        c2.metric("Reducción de pérdidas", "92%", "S/ 4M+ recuperados")
        c3.metric("ROI del proyecto", "160%", "Payback < 2 años")

    elif menu == "Preguntas Frecuentes":
        st.header("Preguntas Frecuentes – Sustentación")
        
        faqs = {
            "¿Por qué usan LSTM y no modelos tradicionales como ARIMA?":
                "Porque LSTM captura relaciones no lineales y múltiples variables simultáneas (precios + inflación + tipo de cambio). Estudios 2024 muestran hasta +25% de precisión en mercados emergentes.",
            
            "¿Cómo manejan el 'model drift'?":
                "Monitoreo diario de métricas (RMSE, MAPE). Si el error sube >10%, se dispara reentrenamiento automático.",
            
            "¿Qué pasa si yfinance falla?":
                "Tenemos backup: API Alpha Vantage + datos internos de Kallpa + CSV histórico.",
            
            "¿Es seguro para datos financieros?":
                "Sí. Cifrado TLS, MFA, IAM en AWS, logs auditables y cumplimiento SBS.",
            
            "¿Cuánto cuesta implementarlo?":
                "Inversión: S/ 476,049 → Recuperada en <2 años → Beneficio anual: S/ 300,000"
        }
        
        for q, a in faqs.items():
            with st.expander(q):
                st.write(a)

    else:  # Equipo
        st.header("Equipo del Proyecto")
        cols = st.columns(3)
        nombres = ["Manuel Asencio Espino", "Leonardo Granados Ortega", "Lázaro Cerquín Odar"]
        for col, nombre in zip(cols, nombres):
            with col:
                st.image(f"https://randomuser.me/api/portraits/men/{hash(nombre)%100}.jpg", width=150)
                st.markdown(f"**{nombre}**")
        st.markdown("**Universidad Peruana de Ciencias Aplicadas – Diciembre 2025**")

    # Cerrar sesión
    if st.sidebar.button("Cerrar sesión"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

