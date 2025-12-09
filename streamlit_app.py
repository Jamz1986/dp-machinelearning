import streamlit as st

st.set_page_config(
    page_title="MVP Machine Learning – JZ4",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MVP – Sistema Predictivo de Machine Learning")
st.markdown("""
Este MVP integra las historias de usuario del proyecto, mostrando un flujo claro 
entre visualización, análisis y predicción. Navega por las páginas en el panel izquierdo.
""")

st.subheader("Componentes incluidos:")
st.markdown("""
- **📊 Visualización de Datos**  
- **🧪 Módulo de Predicción**  
- **🔗 Integración externa opcional (AWS RDS o APIs)**  
""")

st.info("Use el menú lateral para navegar por las páginas del MVP.")
