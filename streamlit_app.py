import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

# -----------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------
st.set_page_config(
    page_title="MVP BVL Predictivo",
    layout="wide",
    page_icon="📈"
)

st.title("📈 MVP – Análisis y Predicción del Mercado Peruano (BVL)")

st.markdown("""
Este prototipo integra **visualización**, **análisis exploratorio** y **predicción algorítmica**
sobre acciones peruanas listadas en la **Bolsa de Valores de Lima (BVL)**.
""")

# -----------------------------
# TABS
# -----------------------------
tabs = st.tabs(["📊 Visualización", "🧮 Modelado Predictivo", "🔮 Predicción"])

# ---------------------------------------------------------
# TAB 1 – VISUALIZACIÓN
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Visualización del Mercado Peruano")

    ticker = st.selectbox(
        "Seleccione un activo de la BVL",
        ["BVN", "CVERDEC1.LM", "CPACASC1.LM", "FERREYC1.LM"]
    )

    data = yf.download(ticker, period="5y")

    if data.empty:
        st.error("No se encontraron datos para el ticker seleccionado.")
    else:
        st.success(f"Datos cargados correctamente: **{ticker}**")

        fig = px.line(data, y="Close", title=f"Precio de Cierre - {ticker}")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Vista rápida del dataset")
        st.dataframe(data.tail())

        data["Return"] = data["Close"].pct_change()
        fig_ret = px.line(data, y="Return", title="Retornos diarios")
        st.plotly_chart(fig_ret, use_container_width=True)

# ---------------------------------------------------------
# TAB 2 – MODELADO
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Entrenamiento del Modelo Predictivo")

    if data.empty:
        st.warning("Primero selecciona un ticker en la sección de Visualización.")
    else:
        df = data.copy()
        df["return"] = df["Close"].pct_change().fillna(0)
        df["ma5"] = df["Close"].rolling(5).mean().fillna(method="bfill")
        df["ma10"] = df["Close"].rolling(10).mean().fillna(method="bfill")

        model_data = df[["Close", "return", "ma5", "ma10"]].dropna()
        model_data["target"] = model_data["Close"].shift(-1)
        model_data = model_data.dropna()

        X = model_data[["Close", "return", "ma5", "ma10"]]
        y = model_data["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = np.mean(np.abs(preds - y_test))
        rmse = np.sqrt(np.mean((preds - y_test) ** 2))

        st.metric("MAE", f"{mae:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")

        st.success("Modelo entrenado correctamente.")

# ---------------------------------------------------------
# TAB 3 – PREDICCIÓN
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Generar Predicción del Siguiente Día")

    if "model" not in globals():
        st.warning("Primero entrena el modelo en la pestaña 'Modelado Predictivo'.")
    else:
        st.info("Ingrese los parámetros para estimar el precio siguiente:")

        col1, col2 = st.columns(2)

        with col1:
            close_in = st.number_input("Close actual", value=float(df["Close"].iloc[-1]))
            ret_in = st.number_input("Return actual", value=float(df["return"].iloc[-1]))

        with col2:
            ma5_in = st.number_input("MA5", value=float(df["ma5"].iloc[-1]))
            ma10_in = st.number_input("MA10", value=float(df["ma10"].iloc[-1]))

        if st.button("Predecir precio"):
            X_pred = np.array([[close_in, ret_in, ma5_in, ma10_in]])
            pred_price = model.predict(X_pred)[0]

            st.success(f"Precio estimado próximo cierre: **{pred_price:.3f}**")

