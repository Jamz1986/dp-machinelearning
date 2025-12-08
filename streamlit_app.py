import streamlit as st
import pandas as pd

st.title('🎈 Kallpa Securities')

st.info('Este es un aplicativo LSTM')

with st.expander('Data'):
   st.write('**Raw Data**')
   df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
   df
